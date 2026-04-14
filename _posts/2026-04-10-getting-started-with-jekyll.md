---
title: "Getting Started with Jekyll"
date: 2026-04-10
description: "A quick intro to building static sites with Jekyll — the setup, the structure, and why I think it's still a great choice."
tags: [jekyll, webdev, tutorial]
---

Static site generators have been around for a while, but Jekyll remains one of the most elegant choices for personal sites and blogs. Here's why I still reach for it.

## Why static?

Static sites are fast, secure, and incredibly cheap to host. No database, no server-side rendering on each request, no attack surface beyond the HTML/CSS/JS you ship.

## The structure

A Jekyll site follows a simple convention:

```
_config.yml      # Site configuration
_layouts/        # HTML templates
_includes/       # Reusable snippets
_posts/          # Blog posts (Markdown)
_sass/           # Sass partials
assets/          # CSS, JS, images
```

## Writing posts

Posts live in `_posts/` and follow the naming convention `YYYY-MM-DD-title.md`. Front matter at the top of each file lets you define metadata:

```yaml
---
title: "My Post"
date: 2026-01-01
tags: [writing, code]
---
```

Everything after the `---` is your Markdown content.

## Deploying

GitHub Pages has built-in Jekyll support. Push your repo and your site builds automatically. Or use Netlify / Vercel for more control over the build process.

Static sites are a joy to work with. Give Jekyll a try.
