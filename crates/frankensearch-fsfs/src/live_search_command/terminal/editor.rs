//! Bounded query editing and latest-request coalescing, independent of ranking.
//!
//! Editing a draft never changes the displayed result identity. Only the owner
//! of a completed subscription poll may take a submitted query and reset both
//! producer and consumer. No query work is queued per keystroke.

use ftui_core::event::{Event, KeyCode, KeyEventKind};

use super::{Action, display_text};
use crate::live_search_command::MAX_QUERY_BYTES;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Command {
    Start,
    Insert(char),
    Paste(String),
    TooLarge,
    Left,
    Right,
    Home,
    End,
    Backspace,
    Delete,
    Clear,
    Submit,
    Cancel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Draft {
    text: String,
    // Always a UTF-8 boundary. Navigation/deletion operate on Unicode scalars.
    cursor: usize,
}

#[derive(Default, Debug)]
pub(super) struct Editor {
    draft: Option<Draft>,
    requested: Option<String>,
    warning: Option<&'static str>,
}

impl Editor {
    pub const fn is_editing(&self) -> bool {
        self.draft.is_some()
    }

    pub fn take_requested(&mut self) -> Option<String> {
        self.requested.take()
    }

    pub fn apply(&mut self, command: Command, active: &str) {
        self.warning = None;
        match command {
            Command::Start => {
                if self.draft.is_none() {
                    let text = self.requested.as_deref().unwrap_or(active).to_owned();
                    self.draft = Some(Draft {
                        cursor: text.len(),
                        text,
                    });
                }
                return;
            }
            Command::Cancel => {
                self.draft = None;
                return;
            }
            Command::TooLarge => {
                self.warning = Some("Query exceeds the 64 KiB limit; text was not inserted");
                return;
            }
            _ => {}
        }
        let Some(draft) = &mut self.draft else {
            return;
        };
        match command {
            Command::Insert(character) => {
                if character.is_control() {
                    self.warning = Some("Control characters are not query input");
                } else if draft.text.len().saturating_add(character.len_utf8()) > MAX_QUERY_BYTES {
                    self.warning = Some("Query exceeds the 64 KiB limit; text was not inserted");
                } else {
                    draft.text.insert(draft.cursor, character);
                    draft.cursor += character.len_utf8();
                }
            }
            Command::Paste(text) => {
                // Reject the whole paste rather than silently search a prefix.
                // The event mapper checks before cloning; recheck here too.
                if text.len() > MAX_QUERY_BYTES {
                    self.warning = Some("Query exceeds the 64 KiB limit; text was not inserted");
                    return;
                }
                if text.chars().any(|character| {
                    character.is_control() && !matches!(character, '\n' | '\r' | '\t')
                }) {
                    self.warning = Some("Paste contains control characters; text was not inserted");
                    return;
                }
                let text: String = text
                    .chars()
                    .map(|character| match character {
                        '\n' | '\r' | '\t' => ' ',
                        other => other,
                    })
                    .collect();
                if draft.text.len().saturating_add(text.len()) > MAX_QUERY_BYTES {
                    self.warning = Some("Query exceeds the 64 KiB limit; text was not inserted");
                    return;
                }
                draft.text.insert_str(draft.cursor, &text);
                draft.cursor += text.len();
            }
            Command::Left => draft.cursor = previous_boundary(draft),
            Command::Right => draft.cursor = next_boundary(draft),
            Command::Home => draft.cursor = 0,
            Command::End => draft.cursor = draft.text.len(),
            Command::Backspace => {
                let previous = previous_boundary(draft);
                draft.text.replace_range(previous..draft.cursor, "");
                draft.cursor = previous;
            }
            Command::Delete => {
                let next = next_boundary(draft);
                draft.text.replace_range(draft.cursor..next, "");
            }
            Command::Clear => {
                draft.text.clear();
                draft.cursor = 0;
            }
            Command::Submit => {
                if draft.text.trim().is_empty() {
                    self.warning = Some("Enter a nonblank query");
                    return;
                }
                if let Some(draft) = self.draft.take() {
                    // Even an identical-to-active query replaces older pending
                    // intent. The subscriber decides whether it is a no-op.
                    self.requested = Some(draft.text);
                }
            }
            Command::Start | Command::Cancel | Command::TooLarge => {}
        }
    }

    /// Bounded, sanitized viewport around the scalar cursor, not the whole draft.
    pub fn prompt(&self, columns: usize) -> Option<String> {
        if let Some(draft) = &self.draft {
            let side = columns.saturating_sub(12) / 4;
            let before = &draft.text[..draft.cursor];
            let after = &draft.text[draft.cursor..];
            let mut left: Vec<char> = before.chars().rev().take(side).collect();
            left.reverse();
            let left: String = left.into_iter().collect();
            let right: String = after.chars().take(side).collect();
            return Some(format!(
                "Query > {}|{}",
                display_text(&left, side),
                display_text(&right, side)
            ));
        }
        self.requested.as_ref().map(|query| {
            format!(
                "Queued query: {} (after current phases)",
                display_text(query, columns.saturating_sub(44))
            )
        })
    }

    pub fn help(&self) -> Option<&str> {
        self.warning.or_else(|| {
            self.is_editing().then_some(
                "Enter: search | Esc: cancel edit | Ctrl-U: clear | arrows/Home/End | Ctrl-C: stop",
            )
        })
    }
}

fn previous_boundary(draft: &Draft) -> usize {
    draft.text[..draft.cursor]
        .char_indices()
        .next_back()
        .map_or(0, |(offset, _)| offset)
}

fn next_boundary(draft: &Draft) -> usize {
    draft.text[draft.cursor..]
        .chars()
        .next()
        .map_or(draft.cursor, |character| {
            draft.cursor + character.len_utf8()
        })
}

pub(super) fn action_for(event: &Event) -> Option<Action> {
    match event {
        Event::Resize { .. } | Event::Focus(true) => Some(Action::Redraw),
        Event::Paste(paste) => Some(Action::Edit(if paste.text.len() > MAX_QUERY_BYTES {
            Command::TooLarge
        } else {
            Command::Paste(paste.text.clone())
        })),
        Event::Key(key) if key.kind != KeyEventKind::Release => {
            if key.ctrl() && key.code == KeyCode::Char('c') {
                return Some(Action::Interrupt);
            }
            if key.ctrl() && !key.alt() && !key.super_key() && key.code == KeyCode::Char('u') {
                return Some(Action::Edit(Command::Clear));
            }
            if key.ctrl() || key.alt() || key.super_key() {
                return None;
            }
            let command = match key.code {
                KeyCode::Enter => Command::Submit,
                KeyCode::Escape => Command::Cancel,
                KeyCode::Char(character) => Command::Insert(character),
                KeyCode::Left => Command::Left,
                KeyCode::Right => Command::Right,
                KeyCode::Home => Command::Home,
                KeyCode::End => Command::End,
                KeyCode::Backspace => Command::Backspace,
                KeyCode::Delete => Command::Delete,
                _ => return None,
            };
            Some(Action::Edit(command))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use ftui_core::event::{KeyEvent, Modifiers, PasteEvent};

    use super::*;

    #[test]
    fn editing_is_utf8_safe_and_never_submits_per_keystroke() {
        let mut editor = Editor::default();
        editor.apply(Command::Start, "aé界");
        editor.apply(Command::Left, "ignored");
        editor.apply(Command::Insert('🦀'), "ignored");
        assert_eq!(editor.draft.as_ref().unwrap().text, "aé🦀界");
        editor.apply(Command::Backspace, "ignored");
        editor.apply(Command::Delete, "ignored");
        assert_eq!(editor.draft.as_ref().unwrap().text, "aé");
        editor.apply(Command::Home, "ignored");
        editor.apply(Command::Backspace, "ignored");
        editor.apply(Command::Right, "ignored");
        editor.apply(Command::Delete, "ignored");
        editor.apply(Command::End, "ignored");
        editor.apply(Command::Right, "ignored");
        assert_eq!(editor.draft.as_ref().unwrap().text, "a");
        assert!(editor.take_requested().is_none());
        editor.apply(Command::Submit, "ignored");
        assert_eq!(editor.take_requested().as_deref(), Some("a"));
        assert!(!editor.is_editing());
    }

    #[test]
    fn submitted_requests_coalesce_and_edit_cancellation_preserves_prior_intent() {
        let mut editor = Editor::default();
        for query in ["beta", "gamma"] {
            editor.apply(Command::Start, "alpha");
            editor.apply(Command::Clear, "alpha");
            editor.apply(Command::Paste(query.to_owned()), "alpha");
            editor.apply(Command::Submit, "alpha");
        }
        editor.apply(Command::Start, "alpha");
        assert_eq!(editor.draft.as_ref().unwrap().text, "gamma");
        editor.apply(Command::Insert('x'), "alpha");
        editor.apply(Command::Cancel, "alpha");
        assert_eq!(editor.take_requested().as_deref(), Some("gamma"));
        assert!(editor.take_requested().is_none());
    }

    #[test]
    fn returning_to_the_active_query_cancels_a_queued_different_query() {
        let mut editor = Editor::default();
        for query in ["beta", "alpha"] {
            editor.apply(Command::Start, "alpha");
            editor.apply(Command::Clear, "alpha");
            editor.apply(Command::Paste(query.to_owned()), "alpha");
            editor.apply(Command::Submit, "alpha");
        }
        assert_eq!(editor.take_requested().as_deref(), Some("alpha"));
    }

    #[test]
    fn blank_and_oversized_drafts_are_not_submitted_or_partially_inserted() {
        let mut editor = Editor::default();
        editor.apply(Command::Start, "alpha");
        editor.apply(Command::Clear, "alpha");
        editor.apply(Command::Submit, "alpha");
        assert!(editor.is_editing());
        assert!(editor.warning.is_some());
        assert!(editor.take_requested().is_none());
        editor.apply(Command::Paste("x".repeat(MAX_QUERY_BYTES - 1)), "alpha");
        let before = editor.draft.clone();
        editor.apply(Command::Insert('é'), "alpha");
        assert_eq!(editor.draft, before);
        editor.apply(Command::Paste("ab".to_owned()), "alpha");
        assert_eq!(editor.draft, before);
        editor.apply(Command::Insert('x'), "alpha");
        assert_eq!(editor.draft.as_ref().unwrap().text.len(), MAX_QUERY_BYTES);
        editor.apply(Command::Submit, "alpha");
        assert_eq!(editor.take_requested().unwrap().len(), MAX_QUERY_BYTES);
    }

    #[test]
    fn paste_is_literal_bounded_input_not_commands_or_terminal_control() {
        let mut editor = Editor::default();
        editor.apply(Command::Start, "");
        editor.apply(Command::Paste("q\nfoo\tbar".to_owned()), "");
        assert_eq!(editor.draft.as_ref().unwrap().text, "q foo bar");
        assert!(editor.take_requested().is_none());
        let before = editor.draft.clone();
        editor.apply(Command::Paste("\x1b[2J".to_owned()), "");
        assert_eq!(editor.draft, before);
        let prompt = editor.prompt(80).unwrap();
        assert!(!prompt.contains('\x1b'));
        assert!(!prompt.contains('\n'));
        assert!(editor.prompt(1).unwrap().contains('|'));
    }

    #[test]
    fn editing_keys_cannot_accidentally_quit_and_releases_are_ignored() {
        let q = KeyEvent::new(KeyCode::Char('q'));
        assert_eq!(
            action_for(&Event::Key(q)),
            Some(Action::Edit(Command::Insert('q')))
        );
        assert_eq!(
            action_for(&Event::Key(KeyEvent::new(KeyCode::Escape))),
            Some(Action::Edit(Command::Cancel))
        );
        assert_eq!(
            action_for(&Event::Key(q.with_kind(KeyEventKind::Release))),
            None
        );
        assert_eq!(
            action_for(&Event::Key(q.with_modifiers(Modifiers::ALT))),
            None
        );
        assert_eq!(
            action_for(&Event::Key(
                KeyEvent::new(KeyCode::Char('c')).with_modifiers(Modifiers::CTRL)
            )),
            Some(Action::Interrupt)
        );
        assert_eq!(
            action_for(&Event::Key(
                KeyEvent::new(KeyCode::Char('u')).with_modifiers(Modifiers::CTRL)
            )),
            Some(Action::Edit(Command::Clear))
        );
        assert_eq!(
            action_for(&Event::Paste(PasteEvent::bracketed(
                "x".repeat(MAX_QUERY_BYTES + 1)
            ))),
            Some(Action::Edit(Command::TooLarge))
        );
    }
}
