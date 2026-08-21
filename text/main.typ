// Bachelor/master thesis -- main file (Typst version).
// Compile with:  typst compile main.typ

#import "ftndiplomski.typ": *

#show: thesis.with(
  author: "Name Surname",
  title: "Thesis Title",
  thesis-type: "Bachelor Thesis",
  studies: "Bachelor Academic Studies",
  place: "Novi Sad",
  year: "2026",
  lang: "en",
)

#title-page(
  author: "Name Surname",
  title: "Thesis Title",
  thesis-type: "Bachelor Thesis",
  studies: "Bachelor Academic Studies",
  place: "Novi Sad",
  year: "2026",
)

// Official forms (thesis assignment sheet, conflict-of-interest
// declaration) are normally bound into the printed copy as a signed,
// scanned PDF rather than retyped. Once signed, insert them like this:
// #set page(numbering: none)
// #image("forms/thesis-assignment.pdf")
// #pagebreak()

#toc(lang: "en")
#start-body()

= Introduction

One or two paragraphs describing the problem and the motivation for
solving it.

A paragraph giving a precise definition of the specific problem
addressed in the thesis.

A paragraph explaining, at a high level of abstraction, how the problem
was solved, so the reader gets a rough picture of the solution -- e.g.
which technologies the solution is based on.

A paragraph explaining what makes the solution unique, i.e. how it
differs from similar solutions.

A paragraph outlining the organization of the rest of the thesis by
chapter.

= Overview of similar systems

This chapter gives a short overview of similar systems/applications and
the criteria used to select them for comparison. Each item below
describes one similar solution, referenced immediately after its name
is introduced, followed by a summary of that solution's strengths and
weaknesses.

= Technologies used

This chapter presents the theoretical background of the technologies
used to develop the application -- for example _Rust_, _Angular_ and
_PostgreSQL_.

== Example technology

A short description of the technology, its key characteristics, and the
reasons it was chosen for this thesis.

= Specification

== System requirements

=== Functional requirements

Description of the functional requirements, if useful illustrated with
a use-case diagram (@use-case).

=== Non-functional requirements

Description of the non-functional requirements (performance, security,
portability, user experience, etc.).

== Data model

Description of the entities and the relationships between them, shown
in the diagram in @data-model.

#figure(
  rect(width: 60%, height: 3cm, stroke: 0.5pt)[
    #align(center + horizon)[Data model diagram placeholder]
  ],
  caption: [Data model diagram],
) <data-model>

#figure(
  rect(width: 60%, height: 3cm, stroke: 0.5pt)[
    #align(center + horizon)[Use-case diagram placeholder]
  ],
  caption: [Use-case diagram],
) <use-case>

= Implementation

== System architecture

Description of all the important elements of the software system's
implementation. Class, method and attribute names in running text use
`monospace` formatting, e.g. class `UserService` or method
`calculateTotalPrice()`.

@rust-example shows an example function written in Rust.

#listing(caption: [Example function in Rust])[
```rust
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```
] <rust-example>

= Demonstration

A walkthrough of the important elements of using the application, step
by step, through one or more scenarios, illustrated with screenshots.

= Conclusion

A recap of the main points of the thesis: the problem solved and the
motivation for solving it, a rough description of the solution, a
comparison with similar solutions, and possible directions for further
extension/improvement.

#unnumbered-chapter([Bibliography])

+ Source name, Author -- #link("https://example.com")

#unnumbered-chapter([Biography])

Name Surname was born on dd.mm.yyyy in City. They completed primary
school "..." and secondary school "...". After finishing secondary
school, they enrolled in the ... study programme at the Faculty of
Technical Sciences in Novi Sad, where they studied from 20XX to 20XX.

#kdi-page(
  author: "Name Surname",
  mentor: "Dr Name Surname, title",
  title: "Thesis Title",
  thesis-type: "Bachelor Thesis",
  language-of-publication: "english",
  year: "2026",
  physical-description: "chapters 7 / pages XX / references X / tables X / illustrations X / graphs X / appendixes X",
  scientific-field: "Software Engineering and Information Technologies",
  scientific-discipline: "Software Engineering",
  keywords: "keyword 1, keyword 2, keyword 3",
  abstract: "One paragraph that captures the essence of the thesis: the problem, the motivation, an outline of the solution and the result.",
  president: "Dr Name Surname, title",
  member: "Dr Name Surname, title",
)

#kwd-page(
  author: "Name Surname",
  mentor: "Name Surname, title, PhD",
  title: "Thesis Title",
  thesis-type: "Bachelor Thesis",
  text-language: "English",
  year: "2026",
  physical-description: "chapters 7 / pages XX / references X / tables X / illustrations X / graphs X / appendixes X",
  scientific-field: "Software Engineering and Information Technologies",
  scientific-discipline: "Software Engineering",
  keywords: "keyword 1, keyword 2, keyword 3",
  abstract: "One paragraph that captures the essence of the thesis: the problem, the motivation, an outline of the solution and the result.",
  president: "Name Surname, full professor, PhD",
  member: "Name Surname, full professor, PhD",
)
