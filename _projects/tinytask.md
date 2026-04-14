---
title: "TinyTask"
description: "A minimal task manager that lives in your terminal. No database, no cloud sync — just a plain-text file you own."
tags: [typescript, cli, productivity]
year: 2026
github: "https://github.com/yourhandle/tinytask"
live: "https://tinytask.sh"
---

TinyTask is a deliberately small task manager. It stores tasks as Markdown in a single file you control. No account required, no data leaves your machine.

## Philosophy

Most productivity tools have too many features. The overhead of managing the tool starts to rival the overhead of managing the work it's supposed to help with.

TinyTask has six commands: `add`, `done`, `list`, `delete`, `move`, and `edit`. That's it.

## Storage format

Tasks are stored in a plain Markdown file:

```markdown
# Tasks

## today
- [ ] Write blog post
- [ ] Review PR #142

## this week
- [ ] Update dependencies
- [x] Fix pagination bug
```

The file is human-readable and editable. Sync it with Dropbox, iCloud, or git. TinyTask doesn't care.

## Implementation

Built with TypeScript and running on Node.js. The CLI uses [`ink`](https://github.com/vadimdemedes/ink) for interactive list views. Parsing is done with a hand-rolled Markdown parser (about 150 lines) rather than a full library, since the format is intentionally simple.
