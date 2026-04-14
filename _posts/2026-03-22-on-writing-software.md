---
title: "On Writing Software"
date: 2026-03-22
description: "Software is a form of writing. The craft principles that make prose clear also make code readable."
tags: [programming, craft]
---

Programming and writing share more DNA than most people acknowledge. Both involve translating ideas into a structured language that others can read. Both reward clarity, concision, and a sense of the audience.

## The reader is the compiler

When you write prose, you have a reader in mind. When you write code, you have two readers: the machine that executes it, and the human who maintains it. The machine is the easier audience. It doesn't care about variable names, structure, or intent. The human cares about all three.

Most bugs I've encountered in production code weren't logic errors in isolation — they were comprehension errors. Someone read code, misunderstood what it did, and introduced a change that made sense given their misunderstanding.

## Clarity as a discipline

Good writers revise. They cut sentences that don't earn their place. They replace vague words with precise ones. They read their work aloud and listen for stumbles.

Good programmers do the same. They refactor not because the code is broken but because it could be clearer. They rename variables. They extract functions. They write the obvious implementation first and then question whether the obvious implementation is actually obvious to someone reading it cold.

## Constraints as a tool

Hemingway's famous iceberg principle: the dignity of movement of an iceberg is due to only one-eighth of it being above water. The other seven-eighths — the research, the cut scenes, the discarded drafts — give the surface its weight.

In code, the equivalent might be the work you delete. Every abstraction you decide *not* to introduce. Every function you collapse back into its call site because the indirection wasn't earning its place.

Writing software well is an act of restraint as much as invention.
