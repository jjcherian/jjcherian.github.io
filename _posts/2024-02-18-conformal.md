---
layout: distill
title: On conformal prediction (Part 1)
date: 2024-02-18 
description: What is conformal prediction and should we care?
tags: conformal prediction
categories: conformal-prediction
giscus_comments: true
featured: false

authors:
  - name: John Cherian

bibliography: 2024-02-18-conformal.bib

toc:
  - name: Introduction
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
#   - name: Citations
#   - name: Footnotes
#   - name: Code Blocks
#   - name: Interactive Plots
#   - name: Layouts
#   - name: Other Typography?
---

## Introduction

Recently, Ben Recht published a series of [blog posts](https://www.argmin.net/p/cover-songs) about conformal prediction. I thought they made some interesting points, and I promised that I would write a blog post about it, i.e., 
{% twitter https://twitter.com/jjcherian/status/1756027416655667313 %}
It turns out the (not so) near future is actually just about two weeks! But this is only part 1 (of 2?)...so maybe my vague forecast will end up being correct after all. 

Let's first motivate conformal prediction with a concrete problem. Imagine we're trying to help a university admissions office admit students who are likely to succeed, so we fit a neural network trained on thousands of historical admissions records and college transcripts to predict a prospective student's final GPA. The ever-astute admissions officer recognizes, however, that these point predictions are far from perfect. So, she asks "can you give me a *range* of GPAs that you can guarantee the student is likely to fall in?" **Conformal prediction sets** solve exactly this task. 

Given the submitted applications ($$X_i$$) and final GPAs ($$Y_i$$) of $$n$$ previous students, the conformal predictor outputs a set for the $$(n + 1)$$-th prospective student satisfying the following **finite-sample** guarantee:

$$\Pr(Y_{n + 1} \in \hat{C}_n(X_{n + 1})) \geq 1 - \alpha.$$

This guarantee is about as clean of a result as you'll see in statistics. The only assumption we had to make was that the $$(n + 1)$$-st student is drawn i.i.d. from the same distribution as the previous $$n$$ students.<d-footnote>We can relax this assumption to exchangeability, but this distinction is not so important for my purposes.</d-footnote>