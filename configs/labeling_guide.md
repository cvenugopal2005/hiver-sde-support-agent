# AppleSupport Intent Labeling Guide

Practical, condensed guide for manually labeling
`data/processed/intent_labeling_sample.csv`. For the full rationale, sample
evidence, and worked derivation behind these rules, see
`configs/intent_taxonomy.md` (Step 4) — this file is the quick-reference
version to use while labeling row by row.

Each row is one AppleSupport conversation thread. `customer_text` is the
first customer message (the one that raises the issue); `conversation_context`
is the full thread transcript in order. **Read the full context before
labeling** — a single message in isolation is often ambiguous.

Fill in exactly one value in the `intent` column per row, using the exact
intent name from the list below. Use `labeling_notes` freely for anything
worth flagging (ambiguity, a second issue present, uncertainty).

## The 8 intents

1. **Battery & Performance** — device/app is slow, freezing, crashing, or
   battery drains/charges abnormally, with no more specific system named.
2. **Connectivity** — WiFi, Bluetooth, or cellular/LTE won't connect or stay
   connected.
3. **Apple Music / iCloud Sync & Data Loss** — content (songs, playlists,
   photos, contacts, backups) fails to sync or disappears.
4. **Account, Purchases & Billing** — login/Apple ID/password/verification
   problems, wrong charges, refunds, subscriptions.
5. **Hardware, Repair & Warranty** — a physical defect, or a repair/
   replacement/warranty request.
6. **Messaging & Communication** — iMessage/SMS/FaceTime/voicemail itself is
   broken (the channel/feature, not the OS keyboard).
7. **Software / App Bug Report** — a specific app/OS feature misbehaves in a
   way not covered by 1-6. Includes the "I" autocorrect glitch (see below).
8. **General / Pre-Purchase Inquiry** — no malfunction described at all;
   a buying, upgrade, or policy question.

## Inclusion / exclusion rules (short form)

| Intent | Belongs | Does NOT belong |
|---|---|---|
| Battery & Performance | freeze/crash/slow/lag/battery drain, no specific subsystem named | perf issue tied to a named subsystem (→ that intent instead) |
| Connectivity | WiFi/BT/cellular/hotspot connection issues | can't sync iCloud/Music (→ #3), can't iMessage/FaceTime (→ #6) |
| Music/Sync & Data Loss | Apple Music/iTunes/playlist/iCloud sync, missing photos/contacts/voicemails | a billing/subscription problem with Apple Music (→ #4) |
| Account/Purchases/Billing | login, Apple ID, password, verification, charges, refunds, subscriptions | a pre-purchase question with no existing account issue (→ #8) |
| Hardware/Repair/Warranty | physical defect, won't power on (hardware framing), warranty/repair/replacement | "battery drains fast" with no hardware-defect/repair language (→ #1) |
| Messaging & Communication | iMessage/SMS/FaceTime/voicemail feature itself broken | the "I" autocorrect glitch (→ #7, even though it happens while typing messages) |
| Software/App Bug Report | any other specific app/OS malfunction; the "I" bug | anything better covered by #1-6 |
| General/Pre-Purchase Inquiry | no problem described, just a buying/policy question | any message that also describes something broken |

## Primary-intent rule (when more than one issue is mentioned)

Label based on the **most specific concrete system or problem named**, not
on what the customer is asking for (almost every message just wants "fix
it," which doesn't discriminate). When a message names more than one
system, apply this fixed priority order — most specific first:

1. Hardware, Repair & Warranty
2. Account, Purchases & Billing
3. Apple Music / iCloud Sync & Data Loss
4. Connectivity
5. Messaging & Communication
6. Battery & Performance
7. Software / App Bug Report
8. General / Pre-Purchase Inquiry (only if NO malfunction is described)

**Fallback when nothing specific is named:** if the message describes some
kind of malfunction but doesn't name a specific subsystem from #2-#5 (e.g.
"fix your update, it's ruined my phone"), label it **Software / App Bug
Report** if it sounds like a bug/quality complaint, or **Battery &
Performance** if it's specifically about speed/battery/freezing language.
If genuinely torn between the two, pick Software/App Bug Report and note it
in `labeling_notes`.

## Ambiguous examples (worked)

- *"my battery is dead and I can't afford AppleCare, do you have
  financing?"* → **Hardware, Repair & Warranty** (repair/financing request
  outranks the battery mention).
- *"since I updated, iCloud photos won't back up"* → **Apple Music/iCloud
  Sync & Data Loss** (specific subsystem named), not Software/App Bug Report.
- *"my texts aren't showing in order since the update"* → **Messaging &
  Communication** (specific feature named), not Software/App Bug Report.
- *"why does my 'I' turn into a question mark box"* → **Software/App Bug
  Report** (the "I" bug), NOT Messaging & Communication, even though it's
  encountered while typing messages.

## The "I" autocorrect bug specifically

This is the well-known iOS 11.1 (Nov 2017) bug where typing the letter "i"
autocorrected into a boxed symbol. **Always label it Software / App Bug
Report**, per the Step 4 decision — it is treated as a sub-case of that
intent, not its own category, even though it's very common in this dataset.
Recognize it by: the customer describing the letter "i"/"I" turning into a
box, question mark, or garbled symbol while typing; or the literal "I️"
character (an "I" with an invisible variation-selector character attached)
appearing in the text.

## How to handle unclear intent

If, after reading the full conversation context, you genuinely cannot tell
which intent applies (not just "it's a bit ambiguous between two adjacent
intents" — genuinely unclear), label the row `UNCLEAR` and explain why in
`labeling_notes`. Do not force a guess into one of the 8 intents just to
fill the cell — `UNCLEAR` rows can be reviewed as a batch afterward, and a
small number of them is expected and fine.

## How to handle multiple issues in one thread

Pick ONE primary intent using the priority-order rule above, based on the
first customer message's stated problem(s). If a second, clearly distinct
issue also appears later in the same thread, keep the single primary-intent
label and note the second issue in `labeling_notes` (e.g. "secondary issue:
also mentions a billing question") rather than inventing a multi-label
scheme — this taxonomy and the classifier it feeds are single-label.

## General labeling hygiene

- Read `conversation_context`, not just `customer_text`, before deciding —
  later messages often clarify what the customer actually meant.
- Non-English messages: label them by the same rules if you can understand
  enough to place them; if not, mark `UNCLEAR` with a note that it's a
  language issue, not a genuine ambiguity.
- Don't infer a resolution outcome from AppleSupport's reply — this
  labeling task is about the CUSTOMER's intent only, not whether the
  conversation was resolved.
