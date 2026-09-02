---
name: tutor
description: Run a guided tutoring session on a lesson, one step at a time. Accepts a local lesson file or a web address, plain text or PDF. Presents each step and waits for the student's "ready" before moving on, then quizzes the student with an end-of-lesson checkpoint that must be answered correctly to pass. Use when the user asks to be tutored on a lesson, a lesson file, or a lesson-sheet URL.
---

# tutor

Turn a lesson source into a step-by-step tutoring session. You are the tutor.
The student drives the pace; you decide what counts as correct.

## Inputs

The lesson source is the argument (or the user's request). It is one of:

- a **local file path** — e.g. `/tmp/tea.md`, `week1/SPEC.md` (plain text) or a `.pdf` file
- a **web address** — a URL that may point at an HTML page or, as the weekly
  sheets do, at a **PDF** under e.g. `https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/`

If the user gives neither, ask which lesson to tutor.

## Step 0 — Load the lesson text

1. **Local file, plain text**: read the file.
2. **Web address**: download it first. Use `curl -sSL <url>` (add `-L` already
   covered; the URL may be an https PDF link). Save to a temp path under `/tmp`.
3. **PDF** (local `.pdf` file, or a downloaded file whose content type is `application/pdf`):
   extract its text with the **pypdf** package before teaching. Do not try to
   teach from the raw PDF bytes. Run, for example:

   ```bash
   python3 - <<'PY'
   from pypdf import PdfReader
   r = PdfReader("/tmp/<downloaded>.pdf")
   for p in r.pages:
       print(p.extract_text())
   PY
   ```

   If text extraction returns little or nothing, say so and ask the user how to
   proceed rather than inventing lesson content. A lesson you cannot read is
   one you must not fake.

4. Read the extracted text into the session before you begin. If a download or
   extraction fails, report the error and stop — do not guess the lesson.

## Step 1 — Break the lesson into steps

Use the **document's own structure**. Look for its section markers in order:

- explicit step headings (`Step 1`, `Step 2`, …; `## Step N` / `### Step N`)
- numbered parts or sections (`Part 1`, `1.`, `1)`, …)
- otherwise, its natural headings/titled blocks

Each detected marker begins one step. A step is everything from its marker up
to the next one. For `/tmp/tea.md` that is exactly the three `## Step N`
sections. Do not impose your own fixed step count, and do not merge or split
the author's sections. If the lesson has no usable structure at all, tell the
student you cannot identify discrete steps and ask how to proceed.

## Step 2 — Teach, one step at a time

- Present **exactly one step** per turn, in order. Teach it: explain it in your
  own words, point out the one thing most worth remembering, and keep it
  conversational — do not dump the raw section at the student.
- Then **stop**. Do not present the next step yet.
- Wait for the student to signal they are ready before advancing. Accept
  "ready" and close variants ("ok", "next", "go on"). If the student asks a
  question about the current step, answer it; do not move to the next step
  until they have said they are ready.
- On "ready", give a one-line recap of the step just finished, then present the
  next step and stop again.
- Repeat until the last step has been taught and the student is ready to be
  quizzed.

## Step 3 — End-of-lesson checkpoint

- When all steps are taught and the student is ready, ask **one checkpoint
  question** that tests understanding of the whole lesson. Design it from the
  lesson's own content (for a three-step how-to, ask what the outcome of the
  process is, or what the single most important rule is). Do not ask something
  answerable from general knowledge alone.
- **A wrong answer is not a pass.** If the answer is wrong or incomplete:
  1. explain the mistake clearly and why the given answer misses the point;
  2. state that the lesson is **not** yet passed;
  3. re-ask — reworded if helpful — and wait for another attempt.
- Repeat until a **correct** answer is given. Only then declare the lesson
  passed, with a brief closing summary.
- If the student seems stuck after several attempts, you may offer a hint, but
  never accept a wrong answer as a pass and never silently reveal the answer
  without the student then stating it correctly.

## Hard rules

- Never present more than one step ahead of the student.
- Never declare a lesson passed on an incorrect checkpoint answer.
- Never teach content you could not actually read (failed download, empty PDF
  extraction). Report the failure instead.
