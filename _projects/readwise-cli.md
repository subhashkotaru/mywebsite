---
title: "Readwise CLI"
description: "A command-line tool for browsing and exporting your Readwise highlights, with fuzzy search and Markdown export."
tags: [go, cli, open-source]
year: 2025
github: "https://github.com/yourhandle/readwise-cli"
---

A terminal-first interface for [Readwise](https://readwise.io), built in Go. Lets you search, filter, and export highlights without leaving the command line.

## Features

- Fuzzy search across all books and articles
- Filter by tag, author, or date range
- Export to Markdown, JSON, or CSV
- Offline mode with local SQLite cache
- Configurable Markdown templates

## Motivation

I spend a lot of time in the terminal, and the Readwise web UI has more friction than I want when I'm trying to quickly find a highlight. Building a CLI meant I could wire it into my note-taking workflow with a few shell aliases.

## Implementation

The interesting part was building the fuzzy search. I used [`go-fuzz`](https://github.com/dvyukov/go-fuzz) to test edge cases and ended up writing a simple scoring function that prioritizes matches at word boundaries.

The local cache is a SQLite database that syncs incrementally — only fetching highlights newer than the last sync timestamp. Cold starts are slow, but after that it's nearly instant.
