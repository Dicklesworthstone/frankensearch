// Included in admission::tests to exercise the real queue, framing and control
// decoder. No model, sleeps, detached workers or deadline expiry are needed.

#[test]
fn abandoned_full_queue_admits_shutdown_without_waiting_for_the_search() {
    run_test_with_cx(|cx| async move {
        let (_directory, listener) = listener();
        let mut intake = Intake::new(Duration::from_secs(60));
        let clients = (0..MAX_PEERS)
            .map(|_| connect(&listener, b"queued query\n"))
            .collect::<Vec<_>>();
        let now = Instant::now();
        for _ in 0..MAX_PEERS.div_ceil(ACCEPTS_PER_TURN) {
            pump(&mut intake, &cx, &listener, now);
        }
        assert_eq!(intake.ready.len(), MAX_PEERS);
        let released = Arc::new(AtomicBool::new(false));
        let owned = Released(Arc::clone(&released));
        let polls = Cell::new(0);
        let mut driving = Box::pin(intake.drive(&cx, &listener, async {
            let _owned = owned;
            poll_fn(|_| {
                polls.set(polls.get() + 1);
                Poll::<()>::Pending
            })
            .await;
        }));
        let mut task = Context::from_waker(Waker::noop());
        assert!(driving.as_mut().poll(&mut task).is_pending());
        assert_eq!(polls.get(), 1);
        drop(clients);
        let mut control_client = connect(&listener, control::SHUTDOWN_REQUEST);
        let Poll::Ready(Ok(DriveOutcome::Shutdown(shutdown))) =
            driving.as_mut().poll(&mut task)
        else {
            panic!("abandoned queue slots must not hide a waiting shutdown"); // ubs:ignore — cfg(test) assertion.
        };
        drop(driving);
        assert_eq!(polls.get(), 1);
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(intake.len(), 0);
        assert!(shutdown.is_shutdown());
        assert_eq!(shutdown.bytes, control::SHUTDOWN_REQUEST);
        // Admission neither sends probe bytes nor claims acknowledgment.
        assert_eq!(
            control_client.read(&mut [0; 1]).unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
        retained_search_checkpoint(&cx).unwrap();
    });
}

#[test]
fn queue_pruning_preserves_live_order_and_write_half_closed_responses() {
    run_test_with_cx(|cx| async move {
        let (_directory, listener) = listener();
        let mut intake = Intake::new(Duration::from_secs(60));
        let mut first = connect(&listener, b"first\n");
        let gone = connect(&listener, b"gone\n");
        let mut second = connect(&listener, b"second\n");
        let closed = connect(&listener, b"closed\n");
        let now = Instant::now();
        pump(&mut intake, &cx, &listener, now);
        assert_eq!(intake.ready.len(), 4);
        first.shutdown(Shutdown::Write).unwrap();
        drop(gone);
        closed.shutdown(Shutdown::Both).unwrap();
        // The same instant rules out expiry as the source of freed slots.
        pump(&mut intake, &cx, &listener, now);
        let frames = intake
            .ready
            .iter()
            .map(|peer| peer.bytes.as_slice())
            .collect::<Vec<_>>();
        assert_eq!(frames, [b"first\n".as_slice(), b"second\n".as_slice()]);
        for client in [&mut first, &mut second] {
            let (mut peer, bytes) = intake
                .next(&cx, &listener, None)
                .await
                .unwrap()
                .unwrap()
                .into_parts();
            peer.write_all(&bytes).unwrap();
            let mut response = vec![0; bytes.len()];
            client.read_exact(&mut response).unwrap();
            assert_eq!(response, bytes);
        }
        assert_eq!(intake.len(), 0);
    });
}

#[test]
fn newly_framed_disconnected_requests_are_not_dispatched() {
    run_test_with_cx(|cx| async move {
        for frame in [
            b"gone\n".as_slice(),
            b"gone without newline".as_slice(),
            control::SHUTDOWN_REQUEST,
        ] {
            let (_directory, listener) = listener();
            let mut intake = Intake::new(Duration::from_secs(60));
            drop(connect(&listener, frame));
            let _healthy = connect(&listener, b"healthy\n");
            pump(&mut intake, &cx, &listener, Instant::now());
            assert!(intake.reading.is_empty());
            assert_eq!(intake.ready.len(), 1);
            let selected = intake.next(&cx, &listener, None).await.unwrap().unwrap();
            assert!(!selected.is_shutdown());
            assert_eq!(selected.bytes, b"healthy\n");
            assert_eq!(intake.len(), 0);
        }
    });
}

#[test]
fn already_disconnected_shutdown_does_not_preempt_live_work() {
    run_test_with_cx(|cx| async move {
        let (_directory, listener) = listener();
        let mut intake = Intake::new(Duration::from_secs(60));
        let client = connect(&listener, control::SHUTDOWN_REQUEST);
        pump(&mut intake, &cx, &listener, Instant::now());
        assert_eq!(intake.ready.len(), 1);
        assert!(intake.ready[0].is_shutdown());
        drop(client);
        let polled = Cell::new(false);
        let result = intake
            .drive(&cx, &listener, async {
                polled.set(true);
                42
            })
            .await
            .unwrap();
        assert!(matches!(result, DriveOutcome::Finished(42)));
        assert!(polled.get());
        assert_eq!(intake.len(), 0);
        retained_search_checkpoint(&cx).unwrap();
    });
}

#[test]
fn write_half_closed_shutdown_still_preempts_and_receives_acknowledgment() {
    run_test_with_cx(|cx| async move {
        let (_directory, listener) = listener();
        let mut intake = Intake::new(Duration::from_secs(60));
        let mut client = connect(&listener, control::SHUTDOWN_REQUEST);
        client.shutdown(Shutdown::Write).unwrap();
        let polled = Cell::new(false);
        let result = intake
            .drive(&cx, &listener, async { polled.set(true) })
            .await
            .unwrap();
        let DriveOutcome::Shutdown(shutdown) = result else {
            panic!("request-side EOF must not revoke a live shutdown consumer"); // ubs:ignore — cfg(test) assertion.
        };
        assert!(!polled.get());
        let (mut peer, bytes) = shutdown.into_parts();
        assert_eq!(bytes, control::SHUTDOWN_REQUEST);
        super::super::write_response(
            &cx,
            &mut peer,
            control::SHUTDOWN_RESPONSE,
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        let mut response = vec![0; control::SHUTDOWN_RESPONSE.len()];
        client.read_exact(&mut response).unwrap();
        assert_eq!(response, control::SHUTDOWN_RESPONSE);
        retained_search_checkpoint(&cx).unwrap();
    });
}

#[test]
fn queued_request_lives_until_its_last_client_handle_closes() {
    run_test_with_cx(|cx| async move {
        let (_directory, listener) = listener();
        let mut intake = Intake::new(Duration::from_secs(60));
        let client = connect(&listener, b"retained\n");
        let retained = client.try_clone().unwrap();
        let now = Instant::now();
        pump(&mut intake, &cx, &listener, now);
        drop(client);
        pump(&mut intake, &cx, &listener, now);
        assert_eq!(intake.ready.len(), 1);
        assert_eq!(intake.ready[0].bytes, b"retained\n");
        drop(retained);
        pump(&mut intake, &cx, &listener, now);
        assert_eq!(intake.len(), 0);
    });
}
