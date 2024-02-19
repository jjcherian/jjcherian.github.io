---
layout: distill
title: On conformal prediction (1/x)
date: 2024-02-18 
description: What is conformal prediction and should we care?
tags: conformal bias-variance haters 
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

Recently, Ben Recht published a series of [blog posts](https://www.argmin.net/p/cover-songs) about conformal prediction. I thought they made some interesting points, but really engaging with his critiques requires more than 140 characters and I'm definitely not going to start paying for X.
{% twitter https://twitter.com/jjcherian/status/1756027416655667313 %}
So as it turns out, the (not so) near future is actually just about two weeks! But this is only part 1 (of 2 posts?)...so maybe my vague forecast will end up being correct after all. 

Anyways, you clicked on this link to learn about conformal prediction, so let's get started by first motivating conformal prediction with a concrete problem. Imagine we're trying to help a university admissions office admit students who are likely to succeed, so we fit a neural network trained on thousands of historical admissions records and college transcripts to predict a prospective student's final GPA. The ever-astute admissions officer recognizes, however, that these point predictions are far from perfect. So, she asks "can you give me a *range* of GPAs that you can guarantee the student is likely to fall in?" 

At first glance, this sounds pretty hard! Predicting a range? Providing a guarantee? Those don't sound like things that neural networks do. But don't worry, if you hand it some data points that were held-out from neural network training, **conformal prediction** will solve exactly this task. Given the submitted applications (we'll denote these by $$X_i$$) and final GPAs (we'll call these $$Y_i$$) of $$n$$ previous students, a conformal predictor makes use of the trained neural network to output a set for the $$(n + 1)$$-st prospective student (referred to as $$\hat{C}_n(X_{n + 1})$$) that is guaranteed to contain the true GPA with some user-specified probability $$1 - \alpha$$.

$$\Pr(Y_{n + 1} \in \hat{C}_n(X_{n + 1})) \geq 1 - \alpha$$

This guarantee is about as clean of a result as you'll see in statistics. It requires only one assumption: the $$(n + 1)$$-st student is drawn i.i.d. from the same distribution as the previous $$n$$ students.<d-footnote>We can relax this assumption to exchangeability, but this won't really matter for our purposes.</d-footnote> Formally, we state this assumption as $$\{(X_i, Y_i)\}_{i = 1}^{n + 1} \stackrel{i.i.d.}{\sim} P$$.

So, why isn't Ben satisfied? He argues that conformal prediction is sweeping two important details under the rug. The guarantee stated above is *marginal* over both the constructed prediction set function $$\hat{C}_n(\cdot)$$ as well as the new students. In plain English, after reading Ben's blog posts, the admissions officer might ask the following questions:

* Even if the prediction set covers the true GPA with high probability over all the applicants, could it be terribly inaccurate for this particular student?
* What if, by chance, the $$n$$ students you used to construct the prediction set were not representative of the data-generating distribution $$P$$? Even if I'm willing to settle for coverage that only holds marginally over the new students, how big could the following coverage deviation possibly be?

$$\Pr(Y_{n + 1} \in \hat{C}_n(X_{n + 1}) \mid \hat{C}_n(\cdot)) - (1 - \alpha)$$

The first question relates to a research area sometimes referred to as ``conditional'' predictive inference. It'll be the topic of a subsequent blog post, but in short, we're never going to be able to guarantee anything about a specific student without making some kind of assumption.<d-footnote>But is object-conditional predictive inference even well-defined? Do I really think that there's a probability distribution over GPAs for any particular student? It's certainly not true if I made $$Y_{n + 1}$$ an image label and $$X_{n + 1}$$ the set of pixels input to a neural network. Anyways, more on this later.</d-footnote>

The second question will be the subject of today's post. 

## Diving deeper

