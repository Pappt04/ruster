// ftndiplomski.typ
//
// Unofficial Typst template for bachelor/master theses at the
// Faculty of Technical Sciences, University of Novi Sad.
//
// Modeled on:
//   - "Šablon za diplomski rad" (the official faculty template, .docx/.pdf)
//   - "Diplomski - Vasilije Milić" (an example of a defended thesis)
//   - "README za autore" (the department's author guidelines)
//
// This is a companion to ftndiplomski.cls (the LaTeX version of the same
// style) -- use whichever toolchain you prefer, the visual result matches.
//
// Compile with:  typst compile main.typ

// ---------------------------------------------------------------------
//  Language strings
//
//  Only a handful of labels change with the thesis language; everything
//  else (body text) is simply whatever you type. The Serbian KDI page
//  keeps its fixed official field labels regardless of `lang`, since
//  that form's structure is set by the university library, not by the
//  language the thesis itself is written in.
// ---------------------------------------------------------------------
#let ftn-strings = (
  en: (
    contents: "Contents",
    bibliography: "Bibliography",
    biography: "Biography",
    figure: "Figure",
    table: "Table",
    listing: "Listing",
  ),
  sr: (
    contents: "Садржај",
    bibliography: "Литература",
    biography: "Биографија",
    figure: "Слика",
    table: "Табела",
    listing: "Листинг",
  ),
)

// ---------------------------------------------------------------------
//  Header with logos (title page + official forms: KDI/KWD)
// ---------------------------------------------------------------------
#let ftn-header(
  university: "University of Novi Sad",
  faculty: "Faculty of Technical Sciences, Novi Sad",
  logo-university: "logo/univerzitet-logo.png",
  logo-faculty: "logo/ftn-logo.png",
) = {
  grid(
    columns: (2cm, 1fr, 2cm),
    align: (center + horizon, center + horizon, center + horizon),
    image(logo-university, height: 1.7cm),
    [
      #university \
      #text(weight: "bold")[#upper(faculty)]
    ],
    image(logo-faculty, height: 1.7cm),
  )
  v(-2pt)
  line(length: 100%, stroke: 0.8pt)
}

// ---------------------------------------------------------------------
//  Title page
// ---------------------------------------------------------------------
#let title-page(
  author: "",
  title: "",
  thesis-type: "Bachelor Thesis",
  studies: "Bachelor Academic Studies",
  place: "Novi Sad",
  year: "2026",
  university: "University of Novi Sad",
  faculty: "Faculty of Technical Sciences, Novi Sad",
  logo-university: "logo/univerzitet-logo.png",
  logo-faculty: "logo/ftn-logo.png",
) = {
  set page(numbering: none)
  ftn-header(
    university: university,
    faculty: faculty,
    logo-university: logo-university,
    logo-faculty: logo-faculty,
  )
  v(2.2cm)
  align(center)[
    #author
    #v(2.6cm)
    #text(weight: "bold", size: 16pt)[#title]
    #v(1.4cm)
    #thesis-type \
    #sym.dash.en #studies #sym.dash.en
  ]
  v(1fr)
  align(center)[#place, #year.]
  pagebreak()
}

// ---------------------------------------------------------------------
//  Key documentation information (Serbian KDI form) /
//  Key Words Documentation (English KWD form)
//
//  Both are official required annexes for the university library
//  catalogue, independent of the language the thesis body is written
//  in -- see the note on `ftn-strings` above.
// ---------------------------------------------------------------------
#let kdi-row(label, value) = table.cell(label) + table.cell(value)

#let kdi-page(
  author: "",
  mentor: "",
  title: "",
  thesis-type: "",
  language-of-publication: "english",
  year: "",
  physical-description: "",
  scientific-field: "",
  scientific-discipline: "",
  keywords: "",
  abstract: "",
  acceptance-date: "",
  defense-date: "",
  president: "",
  member: "",
  university: "University of Novi Sad",
  faculty: "Faculty of Technical Sciences, Novi Sad",
  logo-university: "logo/univerzitet-logo.png",
  logo-faculty: "logo/ftn-logo.png",
) = {
  set page(numbering: none)
  ftn-header(
    university: university,
    faculty: faculty,
    logo-university: logo-university,
    logo-faculty: logo-faculty,
  )
  v(4pt)
  align(center)[*КЉУЧНА ДОКУМЕНТАЦИЈСКА ИНФОРМАЦИЈА*]
  v(4pt)
  set text(size: 9.5pt)
  show table.cell: it => pad(it, y: 3pt)
  table(
    columns: (4.3cm, 1fr),
    stroke: 0.5pt,
    align: left + top,
    [Тип документације, ТД:], [монографска публикација],
    [Тип записа, ТЗ:], [текстуални штампани документ],
    [Врста рада, ВР:], [#thesis-type],
    [Аутор, АУ:], [#author],
    [Ментор, МН:], [#mentor],
    [Наслов рада, НР:], [#title],
    [Језик публикације, ЈП:], [#language-of-publication],
    [Језик извода, ЈИ:], [српски / енглески],
    [Земља публиковања, ЗП:], [Србија],
    [Уже географско подручје, УГП:], [Војводина],
    [Година, ГО:], [#year],
    [Издавач, ИЗ:], [ауторски репринт],
    [Место и адреса, МА:], [Нови Сад, Факултет техничких наука, Трг Доситеја Обрадовића 6],
    [Физички опис рада, ФО:], [#physical-description],
    [Научна област, НО:], [#scientific-field],
    [Научна дисциплина, НД:], [#scientific-discipline],
    [Предметна одредница / кључне речи, ПО:], [#keywords],
    [УДК:], [],
    [Чува се, ЧУ:], [Библиотека Факултета техничких наука, Трг Доситеја Обрадовића 6, Нови Сад],
    [Важна напомена, ВН:], [],
    [Извод, ИЗ:], [#abstract],
    [Датум прихватања теме, ДП:], [#acceptance-date],
    [Датум одбране, ДО:], [#defense-date],
    [Чланови комисије, КО:], [],
    [   председник:], [#president],
    [   члан:], [#member],
    [   члан, ментор:], [#mentor],
  )
  pagebreak()
}

#let kwd-page(
  author: "",
  mentor: "",
  title: "",
  thesis-type: "",
  text-language: "English",
  year: "",
  physical-description: "",
  scientific-field: "",
  scientific-discipline: "",
  keywords: "",
  abstract: "",
  acceptance-date: "",
  defense-date: "",
  president: "",
  member: "",
  university: "University of Novi Sad",
  faculty: "Faculty of Technical Sciences, Novi Sad",
  logo-university: "logo/univerzitet-logo.png",
  logo-faculty: "logo/ftn-logo.png",
) = {
  set page(numbering: none)
  ftn-header(
    university: university,
    faculty: faculty,
    logo-university: logo-university,
    logo-faculty: logo-faculty,
  )
  v(4pt)
  align(center)[*KEY WORDS DOCUMENTATION*]
  v(4pt)
  set text(size: 9.5pt)
  show table.cell: it => pad(it, y: 3pt)
  table(
    columns: (4.3cm, 1fr),
    stroke: 0.5pt,
    align: left + top,
    [Document type, DT:], [monographic publication],
    [Type of record, TR:], [textual material],
    [Contents code, CC:], [#thesis-type],
    [Author, AU:], [#author],
    [Mentor, MN:], [#mentor],
    [Title, TI:], [#title],
    [Language of text, LT:], [#text-language],
    [Language of abstract, LA:], [Serbian / English],
    [Country of publication, CP:], [Serbia],
    [Locality of publication, LP:], [Vojvodina],
    [Publication year, PY:], [#year],
    [Publisher, PB:], [author's reprint],
    [Publication place, PP:], [Novi Sad, Faculty of Technical Sciences, Trg Dositeja Obradovića 6],
    [Physical description, PD:], [#physical-description],
    [Scientific field, SF:], [#scientific-field],
    [Scientific discipline, SD:], [#scientific-discipline],
    [Subject/Keywords, S/KW:], [#keywords],
    [UC:], [],
    [Holding data, HD:], [Library of the Faculty of Technical Sciences, Trg Dositeja Obradovića 6, Novi Sad],
    [Note, N:], [],
    [Abstract, AB:], [#abstract],
    [Accepted by sci. Board on, ASB:], [#acceptance-date],
    [Defended on, DE:], [#defense-date],
    [Defense board, DB:], [],
    [   president:], [#president],
    [   member:], [#member],
    [   member, mentor:], [#mentor],
  )
}

// ---------------------------------------------------------------------
//  Code listings
//
//  The department's guidelines call for a white listing background,
//  syntax-highlighted text, and a caption below (numbered per section,
//  like figures and tables). Typst highlights code natively, so this
//  is just a styled wrapper around a raw block.
// ---------------------------------------------------------------------
// Usage:
//   #listing(caption: [Example function in Rust])[
//   ```rust
//   fn add(a: i32, b: i32) -> i32 { a + b }
//   ```
//   ]
#let listing(body, caption: none) = {
  figure(
    block(
      fill: white,
      stroke: 0.4pt + gray,
      inset: 8pt,
      radius: 2pt,
      width: 100%,
      align(left, body),
    ),
    caption: caption,
    kind: "ftn-listing",
  )
}

// ---------------------------------------------------------------------
//  Main template
// ---------------------------------------------------------------------
#let thesis(
  author: "",
  title: "",
  thesis-type: "Bachelor Thesis",
  studies: "Bachelor Academic Studies",
  place: "Novi Sad",
  year: "2026",
  university: "University of Novi Sad",
  faculty: "Faculty of Technical Sciences, Novi Sad",
  lang: "en",
  logo-university: "logo/univerzitet-logo.png",
  logo-faculty: "logo/ftn-logo.png",
  body,
) = {
  let t = ftn-strings.at(lang)

  set document(title: title, author: author)
  set text(font: "Liberation Serif", size: 12pt, lang: lang)
  set page(
    paper: "a4",
    margin: (left: 3cm, right: 2.5cm, top: 2.5cm, bottom: 2.5cm),
    numbering: none,
    number-align: top + left,
  )
  set par(justify: true, first-line-indent: (amount: 1.25cm, all: true), leading: 0.65em)
  show link: set text(fill: blue)

  let sans = "Liberation Sans"
  set heading(numbering: "1.1.1.1")
  set figure(supplement: none)

  // Chapter (level 1): bold sans, upper-case, right-aligned, "N. TITLE",
  // each starting on a new page -- matches the reference examples.
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
    counter(figure.where(kind: "ftn-listing")).update(0)
    align(right)[
      #text(font: sans, weight: "bold", size: 16pt)[
        #counter(heading).display("1"). #upper(it.body)
      ]
    ]
    v(28pt)
  }

  // Section (level 2): bold sans, sentence case, left-aligned.
  show heading.where(level: 2): it => {
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
    counter(figure.where(kind: "ftn-listing")).update(0)
    v(18pt, weak: true)
    text(font: sans, weight: "bold", size: 13pt)[
      #counter(heading).display("1.1") #it.body
    ]
    v(8pt, weak: true)
  }

  // Subsection (level 3) and sub-subsection (level 4).
  show heading.where(level: 3): it => {
    v(14pt, weak: true)
    text(font: sans, weight: "bold", size: 12pt)[
      #counter(heading).display("1.1.1") #it.body
    ]
    v(6pt, weak: true)
  }
  show heading.where(level: 4): it => {
    v(12pt, weak: true)
    text(font: sans, weight: "bold", style: "italic", size: 12pt)[
      #counter(heading).display("1.1.1.1") #it.body
    ]
    v(4pt, weak: true)
  }

  // Figures/tables/listings numbered per section ("4.2.1" style), with
  // a centered caption below, label ":" separated.
  let sec-numbering(..nums) = {
    let h = counter(heading).get()
    let chap = h.at(0)
    let sec = if h.len() > 1 { h.at(1) } else { 0 }
    [#chap.#sec.#nums.pos().at(0)]
  }
  set figure(numbering: sec-numbering)
  show figure.where(kind: image): set figure(supplement: t.figure)
  show figure.where(kind: table): set figure(supplement: t.table)
  show figure.where(kind: "ftn-listing"): set figure(supplement: t.listing)
  show figure.caption: it => {
    set align(center)
    it
  }

  // Raw code blocks: syntax highlighted, monospace.
  set raw(theme: none)
  show raw: set text(font: "Liberation Mono", size: 9.5pt)

  // Page numbers in the top-left corner, no header rule -- matches the
  // reference examples.
  set page(numbering: "1")

  body
}

// ---------------------------------------------------------------------
//  Table of contents
// ---------------------------------------------------------------------
#let toc(lang: "en") = {
  let t = ftn-strings.at(lang)
  set page(numbering: none)
  align(left)[#text(font: "Liberation Sans", weight: "bold", size: 16pt)[#t.contents]]
  v(12pt)
  outline(title: none, indent: 1.5em)
  pagebreak()
}

// ---------------------------------------------------------------------
//  Start the (numbered) body of the thesis at page 1.
// ---------------------------------------------------------------------
#let start-body() = {
  set page(numbering: "1")
  counter(page).update(1)
}

// ---------------------------------------------------------------------
//  Bibliography / Biography headings (unnumbered chapters), matching
//  the style of numbered chapters but without a chapter number.
// ---------------------------------------------------------------------
#let unnumbered-chapter(title) = {
  pagebreak(weak: true)
  align(right)[
    #text(font: "Liberation Sans", weight: "bold", size: 16pt)[#upper(title)]
  ]
  v(28pt)
  [#metadata(title) <ftn-unnumbered-chapter>]
}
