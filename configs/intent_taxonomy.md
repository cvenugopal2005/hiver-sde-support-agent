# AppleSupport Intent Taxonomy (Step 4)

Status: **draft taxonomy for the golden set and classifier, not yet used to
build one.** Derived from the keyword/manual analysis in
`scripts/explore_applesupport.py` (sample seed=42) and
`scripts/validate_applesupport.py` (sample seed=123, independent
replication). All counts below are from those two 400-thread samples of
AppleSupport conversations, not the full ~80k-thread AppleSupport
population. They are viability signals, not population-level claims.

## How this taxonomy was derived

Step 3A proposed 8 candidate groups from manual reading of 677 real customer
messages. Step 3B replicated the analysis on an independent 627-message
sample and found every candidate group recurred at a similar rate. Step 4
(this file) merges/splits those candidates into a final 8-intent taxonomy,
explicitly resolves the "I" autocorrect bug question, and defines a
consistent primary-intent rule.

Two candidates from Step 3A were merged going into Step 4:
- `music_sync_dataloss` (was two separate Step 3A clusters: Apple
  Music/iTunes sync, and general data loss) — kept merged, since in practice
  almost every data-loss message in the samples was music/photo/contact sync
  related, not a separate mechanism.
- `account_purchase_billing` (was two separate Step 3A clusters: account/
  Apple ID/login, and purchases/billing/subscriptions) — kept merged, since
  both are non-technical "we need to verify who you are / your payment"
  conversations that behave the same way operationally (DM handoff for
  identity/payment details).

One candidate was added going into Step 4 that was NOT in the original
8-item list the assignment enumerated: **General / Pre-Purchase Inquiry**.
Step 3A's manual reading turned up a small but real cluster of messages that
describe no malfunction at all (e.g. "can I buy an unlocked iPhone X and use
it in Saudi Arabia?"). This is kept as its own intent (see "Why 8 intents"
below) even though the original enumerated list didn't name it, because it
behaves completely differently for the auto-handle/escalate decision that is
Step 6 of this assignment.

## Why 8 intents (not fewer, not more)

- Fewer would force genuinely different resolution paths (e.g. "my phone
  won't turn on" vs. "can I buy AppleCare") into one bucket, which would make
  the auto-handle/escalate decision and reply grounding worse, not simpler.
- More would fragment the samples we have — several candidates
  (Messaging/FaceTime, Hardware/Repair/Warranty, General Inquiry) are already
  the thinnest classes at ~1-2% of a 400-thread sample; splitting further is
  not justified by the evidence in hand.

## Final intents

### 1. Battery & Performance

**Definition:** the device or an app is slow, freezing, crashing, or the
battery drains/charges abnormally — usually described as starting after an
iOS update.

**Belongs:** battery drain/charging complaints; freezing, crashing, hanging,
lagging, general "my phone is slow" reports; explicitly attributed to "since
the update" without naming a more specific subsystem (WiFi, Music, etc.).

**Does NOT belong:** performance complaints that name a specific broken
subsystem (e.g. "since the update my WiFi keeps disconnecting" → Connectivity,
not this intent — see primary-intent rule below); battery/charging framed as
a hardware defect claim ("my battery is swollen / phone won't turn on at all,
Apple Store says it's a hardware issue" → Hardware/Repair/Warranty).

**Real examples (from the samples):**
- "@115858 my battery life sucks after the update????"
- "...my iPhone 6s+ freezes and apps are unresponsive. Latest update doesn't fix the problem. Keep working!!!"
- "@AppleSupport my iphone x battery percentage worked perfectly well before upgrading to ios 11.1, now the percentage seems to get stuck at 80%..."

**Approximate sample evidence:** 88/677 (13.0%) in sample1, 86/627 (13.7%) in
sample2 by the primary-intent keyword rule below — the single largest or
second-largest intent in both samples.

**Grounding usefulness:** **Weak.** Public replies for this intent are
overwhelmingly clarifying questions ("Which iPhone do you have? Let us know
in DM") or DM handoffs, per the Step 3B reply classification (DM handoff was
~51% of ALL AppleSupport replies across both samples, and battery/perf
threads make up a large share of that). Few threads show a visible, reusable
public fix.

---

### 2. Connectivity (Wi-Fi / Bluetooth / Cellular)

**Definition:** the device cannot connect to, stay connected to, or properly
use WiFi, Bluetooth, cellular data, or LTE/hotspot features.

**Belongs:** WiFi drops/won't turn off, Bluetooth pairing failures, cellular
data / LTE issues, personal hotspot problems.

**Does NOT belong:** connectivity issues framed purely as "can't sync
iCloud/Music" (→ Apple Music/iCloud Sync) or "can't send iMessage/FaceTime"
(→ Messaging & Communication), where the network layer isn't what the
customer is complaining about.

**Real examples:**
- "why won't my personal hotspot connect to my laptop? I've tried restarting my network settings on my iphone and restarting my laptop but it still doesn't work."
- "Yo @AppleSupport my watch hasn't been connecting to WiFi lately. Help?"
- "@AppleSupport why can't I turn WiFi & Bluetooth off from control center like I could before?"

**Approximate sample evidence:** 27/677 (4.0%) sample1, 18/627 (2.9%) sample2.
Smaller class; still present in both independent samples.

**Grounding usefulness:** **Moderate.** Some direct troubleshooting is
visible in public replies (e.g. "Have you tried restarting your iPhone?");
still frequently ends in a DM handoff link for account-specific network
diagnostics.

---

### 3. Apple Music / iCloud Sync & Data Loss

**Definition:** content (songs, playlists, photos, contacts, backups) fails
to sync, or disappears/is lost across devices/iCloud.

**Belongs:** Apple Music/iTunes library or playlist problems, iCloud sync
failures, missing photos/contacts/voicemails/notes, backup/restore questions.

**Does NOT belong:** a subscription/payment problem with Apple Music (→
Account, Purchases & Billing, since the issue is billing, not sync); a
message-app-specific data issue (e.g. "my texts aren't showing in order" →
Messaging & Communication).

**Real examples:**
- "@115948 @AppleSupport 60 something songs just disappeared from my key playlist from Apple Music ON THEIR OWN. I didn't remove any song"
- "I'm so stressed I started crying because iTunes refuses to sync songs with my iPod as per usual"
- "why have all my voicemails disappeared? I had saved voicemails from 2 years ago that meant a lot to me and now they've gone."

**Approximate sample evidence:** 34/677 (5.0%) sample1, 34/627 (5.4%)
sample2 — one of the most stable counts across the two samples.

**Grounding usefulness:** **Mixed.** Some replies give a concrete pointer
("here's how backups work: [link]"); notably, in at least one sampled
thread the customer resolved their own playlist-loss problem by exporting/
re-importing an XML file — a customer-authored fix, not an AppleSupport
answer. Any retrieval-based grounding must not conflate "the customer said
it's resolved" with "AppleSupport supplied the resolution."

---

### 4. Account, Purchases & Billing

**Definition:** anything about signing in, Apple ID/iCloud password/
verification, subscriptions, refunds, or being charged incorrectly.

**Belongs:** login/password/verification failures, suspicious/unauthorized
charges, subscription cancellation/refund requests, payment method declined.

**Does NOT belong:** a pre-purchase question with no existing account problem
(→ General/Pre-Purchase Inquiry); a data-loss complaint that happens to
mention iCloud only as the storage location, with no billing/login angle (→
Apple Music/iCloud Sync & Data Loss).

**Real examples:**
- "@AppleSupport why my credit card keep declined by itunes?"
- "@AppleSupport I didnt want to re-purchase [...] I got charged today. Please help to unsubcribe and get refund."
- "Hello @AppleSupport Impossible since 3 days to access my purchase history, ask me for pwd 4 times and come back to itunes. pwd is correct :("

**Approximate sample evidence:** 28/677 (4.1%) sample1, 23/627 (3.7%)
sample2.

**Grounding usefulness:** **Weak by design.** Identity- and payment-sensitive
issues cannot be resolved in public replies; expect a very high DM-handoff
rate for this intent specifically (consistent with the ~51% overall DM-handoff
rate measured in Step 3B). Auto-handling should treat this intent
conservatively regardless of classifier confidence.

---

### 5. Hardware, Repair & Warranty

**Definition:** a physical defect, damage, or out-of-warranty situation
where the customer wants a repair, replacement, or financing option.

**Belongs:** device won't power on/charge (hardware, not software, framing),
warranty disputes, Genius Bar/Apple Store repair experiences, physical damage.

**Does NOT belong:** "my battery drains fast" without any hardware-defect or
repair/warranty language (→ Battery & Performance).

**Real examples:**
- "@AppleSupport It's died - Apple Store confirmed a hardware issue they want to charge me £330 to repair - or I could buy a new one..."
- "@AppleSupport my phone is out of warranty and needs replacement. Do you have any financing options for replacement phones?"
- "@AppleSupport HiSierra caused Mac to crash. Apple store flashed OS & charged. In 2 days, wifi failed. Apple store removed HiSierra and reverted to Sierra."

**Approximate sample evidence:** 16/677 (2.4%) sample1, 9/627 (1.4%) sample2
— the thinnest of the "clearly distinct" intents; the count nearly halved
between the two samples, which is a real signal to watch, not just noise to
ignore.

**Grounding usefulness:** **Weak.** These conversations mostly reference
in-person store visits or phone calls that happened outside Twitter — the
public text is a complaint/narrative, not a resolution, and there is little
reusable canned language to ground a reply in.

**Viability flag:** this is the smallest class with real doubt about whether
it holds up at golden-set scale. Recommend a full-dataset (not just
400-thread sample) keyword count for this intent specifically before
finalizing golden-set quotas — not done in this step to avoid processing
more data than necessary before the taxonomy itself is agreed.

---

### 6. Messaging & Communication (iMessage / SMS / FaceTime / Voicemail)

**Definition:** a messaging or calling feature itself is broken — sending,
receiving, ordering, or transcribing messages/calls.

**Belongs:** iMessage/SMS delivery or ordering bugs, FaceTime call/voicemail
failures, voicemail transcription issues.

**Does NOT belong:** the "I" autocorrect glitch, even though it manifests
while typing messages (→ Software/App Bug Report; see dedicated section
below) — this intent is about the communication channel/feature failing to
work, not a text-input rendering bug.

**Real examples:**
- "Why can't you leave FaceTime voicemails? @115858 get on this shit!!"
- "@AppleSupport I'm receiving duplicate messages from android users and I can't figure it out."
- "@AppleSupport iPhone 6S with iOS 11.03 - no transcription of voicemails, texts message won't scroll. Help."

**Approximate sample evidence:** 9/677 (1.3%) sample1, 8/627 (1.3%) sample2
— smallest class, but the count is nearly identical across both independent
samples, which is reassuring for its stability even at low volume.

**Grounding usefulness:** **Weak-to-moderate.** Similar pattern to
Connectivity — occasional direct troubleshooting question, frequent DM
handoff.

**Viability flag:** same caution as Hardware/Repair/Warranty — thin sample
counts. Kept as its own intent for this draft because it is a functionally
distinct customer need (channel doesn't work vs. device is slow/broken), but
flagged as the most likely candidate to merge into Software/App Bug Report if
full-dataset counts don't support a standalone class.

---

### 7. Software / App Bug Report (includes the "I" autocorrect bug)

**Definition:** a specific app or OS feature misbehaves in a way not covered
by the more specific intents above (not battery/perf, not connectivity, not
sync/data-loss, not messaging, not billing, not hardware) — the generic
"something in the software is broken" bucket, including keyboard/autocorrect
bugs.

**Belongs:** app crashes/misbehavior tied to a specific feature (e.g. share
menu greyed out, Siri misfiring, Safari cookie bug); keyboard/autocorrect
bugs generally; the iOS 11.1 "I"-autocorrect glitch specifically (see below).

**Does NOT belong:** anything better described by a more specific intent
above (that specificity takes priority — see the primary-intent rule).

**Real examples:**
- "@AppleSupport why does my 'I️' turn into 'A [?]'?" (the "I" bug)
- "@AppleSupport why is the share menu greyed out? iOS 11.1"
- "@115858 CAN YOU FRICKING FIX THIS BUG ALREADY"

**Approximate sample evidence (union of generic bug mentions and the "I" bug
sub-case):** 123/677 (18.2%) sample1, 111/627 (17.7%) sample2 — the largest
intent in the taxonomy once merged.

**Grounding usefulness:** **Strong for the "I"-bug sub-case, weak for the
rest.** See the dedicated decision below.

---

### 8. General / Pre-Purchase Inquiry

**Definition:** the customer describes no malfunction at all — a question
about buying, upgrading, or a policy, asked before or independent of any
problem.

**Belongs:** "can I buy X and use it in country Y", "how long can I add
AppleCare after purchase", "can I upgrade my MacBook's RAM through Genius".

**Does NOT belong:** any message that also describes something broken, even
if it also contains a question (the malfunction takes priority — see rule
below).

**Real examples:**
- "@AppleSupport! What If I buy Iphone x(unlocked) from usa? Will I get the warranty? In India? #urgenthelp!"
- "Does anyone know how long you've got to put apple care on your phone ??? Or @AppleSupport can you answer the question?"
- "@AppleSupport Would it be possible to get a macbook pro's ram and SSD upgraded through genius after I purchase it? (Not for free)"

**Approximate sample evidence:** a narrow keyword regex found only ~4 in
sample1, but Step 3A's manual reading of the same sample found roughly 10
messages of this type — **the keyword proxy clearly undercounts this
intent**, because these questions are phrased too variably for a short
keyword list. This is the one intent where we do not trust the automated
count; it should be sampled/labeled manually for the golden set rather than
keyword-filtered.

**Grounding usefulness:** **Different kind of "useful."** These are
informational, not troubleshooting, so grounding means retrieving Apple's
stated policy/link, not a historical fix. Likely the easiest intent to
auto-handle correctly, and the one where reply grounding is most tractable
precisely because there's no account-specific state involved.

## The "other / uncategorized" residual (a limitation, not a 9th intent)

Applying the keyword-based primary-intent rule (below) to full messages,
53.5% (sample1) / 56.6% (sample2) of customer messages did not match any
intent's keyword pattern — mostly short, generic complaints ("fix your
update", "my phone is messed up") with no specific noun a keyword regex can
catch. **This is a limitation of the cheap keyword proxy used for viability
checking here, not evidence that half of AppleSupport's traffic has no
intent.** A human labeler (or later an LLM classifier) reading full message
context will resolve most of this residual into Battery & Performance or
Software/App Bug Report (the two broadest "something is wrong" intents) using
the fallback rule below. The 8 intents above and their approximate counts
should be read as a floor, not a ceiling, on class size.

## Primary-intent rule

**Rule: primary intent = the most specific concrete system or problem the
message names, not the customer's requested action.** Almost every message's
implicit requested action is "fix it" or "help me," which does not
discriminate between intents — so action-based labeling was rejected as
unworkable. The problem being described discriminates much better and is
what determines which historical replies are actually relevant to ground a
response in.

When a message names more than one system (e.g. "since the update my
battery AND wifi are both broken"), apply this fixed priority order — most
specific/least ambiguous first:

1. Hardware, Repair & Warranty
2. Account, Purchases & Billing
3. Apple Music / iCloud Sync & Data Loss
4. Connectivity
5. Messaging & Communication
6. Battery & Performance
7. Software / App Bug Report
8. General / Pre-Purchase Inquiry (only if NO malfunction is described at all)

Rationale for this order: intents 1-5 name a specific subsystem or business
process, so if any of their keywords/topics are present, they are almost
certainly the real issue even if battery/performance language also appears
as a side effect. Battery & Performance and Software/App Bug Report are the
generic "something is broken" catch-alls and are checked last, in that order,
because they are the most likely to spuriously co-occur with a more specific
complaint. General/Pre-Purchase Inquiry is the fallback of last resort and
only applies when literally no malfunction is described.

**Fallback for the "other/uncategorized" residual:** if, after reading full
context, no intent 1-5 applies and no clear malfunction subsystem is named,
default to Software / App Bug Report if any malfunction is described (even
vaguely — "it's ruined", "it sucks now"), or General / Pre-Purchase Inquiry
if no malfunction is described at all.

## Ambiguity rules (worked examples)

- *"my battery is dead and I can't afford AppleCare, do you have financing?"*
  → Hardware, Repair & Warranty (financing/replacement request is priority 1,
  outranks the battery/performance mention).
- *"since I updated, iCloud photos won't back up"* → Apple Music/iCloud Sync
  & Data Loss (priority 3 names the specific subsystem), not Software/App
  Bug Report.
- *"my texts aren't showing in order since the update"* → Messaging &
  Communication (priority 5), not Software/App Bug Report, because a
  specific communication feature is named.
- *"why does my 'I' turn into a question mark box"* → Software / App Bug
  Report (the "I" bug sub-case), NOT Messaging & Communication, even though
  it is encountered while typing messages — see next section.

## The "I" autocorrect bug: decision

**Decision: (B) — include it as a labeled sub-case within Software / App Bug
Report, not as its own top-level intent, and not excluded.**

Evidence considered:
- **Volume is large and stable**: ~88/677 (13.0%) and ~79/627 (12.6%)
  of customer messages across the two independent samples — the
  second-largest single pattern found, after only the merged Battery &
  Performance intent.
- **Grounding is unusually strong for this specific sub-case**: 44/536
  (8.2%) and 35/520 (6.7%) of ALL AppleSupport replies across the two
  samples are a nearly identical canned sentence ("Here's what you can do to
  work around the issue until it's fixed in a future software update:
  [link]") — the single best-grounded, most reusable historical response in
  either sample.
- **But it looks like one dated incident, not a durable intent category**:
  it is tied to a specific, narrow software version (iOS 11.1, Oct-Nov 2017),
  has an unusually uniform text signature (the literal "I️" variation-selector
  character recurring verbatim across hundreds of tweets), and an unusually
  uniform canned reply — all consistent with a single viral bug report wave,
  not a recurring category of diverse customer problems the way "battery
  drains" or "WiFi won't connect" are.

**Why not (A) — its own top-level intent:** would optimize the taxonomy for
a one-time, time-boxed anomaly in this specific dataset window rather than a
generalizable support category. If asked to defend "why does your 5-10
intent taxonomy include a dedicated class for one November 2017 keyboard
bug," there is no good answer — it doesn't reflect how a real support
taxonomy should be designed.

**Why not (C) — exclude/drop it:** it is real historical data and represents
genuine historical AppleSupport behavior including the best example of a
clean, reusable, fully-public resolution available anywhere in the sample.
Dropping it would throw away ~13% of valid Software/App Bug Report training
signal and the single strongest piece of evidence for the reply-grounding
part of this assignment.

**Consequence flagged for later steps:** because it is folded into Software
/ App Bug Report, that intent's aggregate reply-quality/grounding metrics
will look disproportionately good, propped up by this one sub-case. This
belongs explicitly in the "what is misleading about my headline number"
section of the final report (a later step, not this one).

## Evidence and limitations of this taxonomy (Step 4 specific)

- All counts are from two independent ~400-thread / ~650-message samples
  (seeds 42 and 123) of AppleSupport threads, not the full ~80k-thread
  AppleSupport population — treat every number here as a sample-level
  estimate, not a population claim.
- The primary-intent classification used to produce the counts above is a
  simple keyword/regex proxy (see `scripts/validate_applesupport.py`'s
  pattern dictionary), used here only to sanity-check class sizes — it is
  NOT the labeling method to be used for the golden set, which should involve
  a human (or LLM-assisted human-reviewed) reading of full message context.
- General/Pre-Purchase Inquiry's true size is known to be undercounted by
  the keyword proxy (Step 3A manual reading found ~2.5x more examples than
  the regex did) — this intent needs manual sampling, not keyword filtering,
  when the golden set is built.
- Hardware/Repair/Warranty and Messaging/Communication are the thinnest
  classes (1-2% of sample) and their counts moved by up to ~2x between the
  two samples — worth a full-dataset volume check before committing to
  golden-set quotas for these two specifically.
- No non-English message handling.
- This taxonomy has not yet been used to label any data; no golden
  evaluation set exists yet (explicitly out of scope for this step).
