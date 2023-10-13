---
theme: gaia
transition: fade
style: |
  /* ⬇️ Mark the image of "1" in every pages as morphable image named as "one" ⬇️ */
  img[alt="1"] {
    view-transition-name: one;
    contain: layout;
  }

  /* Generic image styling for number icons */
  img:is([alt="1"], [alt="2"], [alt="3"]) {
    height: 64px;
    position: relative;
    top: -0.1em;
    vertical-align: middle;
    width: 64px;
  }
---

# Today's topics

- ![1](https://icongr.am/material/numeric-1-circle.svg?color=666666) Introduction
- ![2](https://icongr.am/material/numeric-2-circle.svg?color=666666) Features
- ![3](https://icongr.am/material/numeric-3-circle.svg?color=666666) Conclusion

---

<!-- _class: lead -->

![1 w:256 h:256](https://icongr.am/material/numeric-1-circle.svg?color=ff9900)

# Introduction

---

# ![1](https://icongr.am/material/numeric-1-circle.svg?color=666666) Introduction

Marp is an open-sourced Markdown presentation ecosystem.

<!-- backgroundImage: "linear-gradient(to bottom, #67b8e3, #0288d1)" -->
<!--
_backgroundColor: black
_color: white
-->


---
![bg left:40% 80%](https://raw.githubusercontent.com/marp-team/marp/master/marp.png)

# **Marp**

Markdown Presentation Ecosystem

https://marp.app/

---

# How to write slides

Split pages by horizontal ruler (`---`). It's very simple! :satisfied:

```markdown
# Slide 1

foobar

---

# Slide 2

foobar
```
