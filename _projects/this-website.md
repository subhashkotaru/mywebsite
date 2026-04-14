---
title: "This Website"
description: "My personal site, built with Jekyll. Minimal design with dark mode support, zero JavaScript dependencies."
tags: [jekyll, css, design]
year: 2026
github: "https://github.com/yourhandle/yourhandle.github.io"
live: "https://yourname.com"
---

This site is built with Jekyll, hosted on GitHub Pages, and designed with the goal of being as fast and readable as possible.

## Design decisions

- System font stack for performance
- No CSS frameworks or utility libraries — all styles are hand-written Sass
- Dark mode via CSS custom properties and `data-theme` attribute
- Zero JS dependencies — the only script is the theme toggle (~40 lines)

## Performance

Lighthouse scores:
- Performance: 100
- Accessibility: 100
- Best Practices: 100
- SEO: 100

## Typography

Body text is set in DM Sans at 16px with a line height of 1.7. Code blocks use DM Mono. Both are loaded from Google Fonts with `font-display: swap` to avoid layout shift.
