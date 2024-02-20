---
layout: distill
title: On conformal prediction
date: 2024-02-18 
description: What is conformal prediction and why should we care about it?
tags: conformal bias-variance 
categories: conformal-prediction
giscus_comments: true
featured: true

authors:
  - name: John Cherian

bibliography: 2024-02-18-conformal.bib

toc:
  - name: Introduction
  - name: What is conformal prediction doing?
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
So as it turns out, the (not so) near future is actually just about three weeks!

Anyways, you clicked on this link to learn about conformal prediction, so let's get started by first motivating conformal prediction with a concrete problem. Imagine we're trying to help a university admissions office admit students who are likely to succeed, so we fit a neural network trained on thousands of historical admissions records and college transcripts to predict a prospective student's final GPA. The ever-astute admissions officer recognizes, however, that these point predictions are far from perfect. So, she asks "can you give me a *range* of GPAs that you can guarantee the student is likely to fall in?" 

At first glance, this sounds pretty hard! Predicting a range? Providing a guarantee? Those don't sound like things that neural networks do. But don't worry, if you hand it some data points that were held-out from neural network training, (split) **conformal prediction** will solve exactly this task. Given the submitted applications (we'll denote these by $$X_i$$) and final GPAs (we'll call these $$Y_i$$) of $$n$$ previous students, a conformal predictor makes use of the trained neural network to output a set for the $$(n + 1)$$-st prospective student (referred to as $$\hat{C}_n(X_{n + 1})$$) that is guaranteed to contain the true GPA with some user-specified probability $$1 - \alpha$$.

$$\Pr(Y_{n + 1} \in \hat{C}_n(X_{n + 1})) \geq 1 - \alpha$$

This guarantee is about as clean of a result as you'll see in statistics. It requires only one assumption: the $$(n + 1)$$-st student is drawn i.i.d. from the same distribution as the previous $$n$$ students.<d-footnote>We can relax this assumption to exchangeability, but this won't really matter for our purposes.</d-footnote>

So, why isn't Ben satisfied? He argues that conformal prediction is sweeping two important details under the rug. The guarantee stated above is *marginal* over both the random data set used to construct $$\hat{C}_n(\cdot)$$ as well as the applicant. In plain English, after reading Ben's blog posts, the admissions officer might ask the following questions:

* Even if the prediction set covers the true GPA with high probability over a random new applicant, could it be terribly inaccurate for this particular student?
* Even if I'm willing to settle for coverage that only holds marginally over a random applicant, what if, by chance, the $$n$$ students you used to construct the prediction set were not representative of the data-generating distribution $$P$$? In that case, how big could the following deviation in the realized coverage possibly be?

$$D_n := \Pr(Y_{n + 1} \in \hat{C}_n(X_{n + 1}) \mid \hat{C}_n(\cdot)) - (1 - \alpha)$$

The first question relates to a research area sometimes referred to as ``conditional'' predictive inference. It'll be the topic of a subsequent blog post.<d-footnote>As a side note, is object-conditional predictive inference even what we want? It's certainly unclear for many ML problems. Imagine if I defined Y to be an image label and X to be the set of pixels input to a neural network. What's random about Y given X? Anyways, more on this later.</d-footnote>

The second question will be the subject of today's post, but to come up with a satisfying answer, we're going to have describe what conformal prediction actually does.

## What is conformal prediction doing?

While there are many methods that fall under the conformal prediction (CP) umbrella, the most popular approach is split CP. In split CP, we use the hold-out data set to construct a set of "conformity scores." The extremely naive choice for this conformity score that we will use as a running example is the absolute value of the observed residual,

$$S(X_i, Y_i) := |Y_i - f(X_i)|.$$

Then, the set $$\hat{C}_n(X_{n + 1})$$ is the set of all $y$ such that $$S(X_{n + 1}, y)$$ is less than some well-chosen threshold $$\hat{q}$$. Mathematically, conformal prediction transforms the prediction set membership problem into a simple quantile estimation problem:

$$\Pr(Y_{n + 1} \in \hat{C}_n(X_{n + 1})) = \Pr(S(X_{n + 1}, Y_{n + 1}) \leq \hat{q}) \stackrel{?}{=} 1 - \alpha.$$

In split conformal prediction, we take $$q^*$$ to be the $$\frac{\lceil (1 - \alpha) \cdot (n + 1) \rceil}{n}$$ quantile of the $$n$$ observed conformity scores. Putting all of this together, we define the split conformal prediction set as

$$ \hat{C}_n(X_{n + 1}) = \left \{y : S(X_{n + 1}, y) \leq \text{Quantile}\left(\frac{\lceil (1 - \alpha) \cdot (n + 1) \rceil}{n}, \{S(X_i, Y_i)\}_{i = 1}^n \right) \right\}.$$

Parsing the definition of this prediction set to prove the coverage result above isn't difficult, but it's easy to lose the forest for the trees. To understand what's happening, let's take a step back and think about what we might have done to construct a prediction set if no one had ever told us about split CP. 

Given a pretty good point predictor $$f(\cdot)$$, the most natural construction for a prediction set is to simply add and subtract a constant to the prediction, e.g.,

$$
\hat{C}^{naive}_n(X_{n + 1}) = [f(X_{n + 1}) - \epsilon, f(X_{n + 1}) + \epsilon].
$$

Intuitively, for this prediction set to be valid, we need $$\epsilon$$ to be larger than $$(1 - \alpha)$$-% of the errors we expect to see on out-of-sample data. Lucky for us, we already obtained a hold-out set that consists of exactly $$n$$ such out-of-sample errors. So, here's a naive strategy: what if we just used the empirical $$(1 - \alpha)$$ quantile of those values? This method for choosing $$\epsilon$$ yields a prediction set *nearly* identical to the split CP $$\hat{C}_n(\cdot)$$. Hopefully, you can convince yourself that for this choice of $$\epsilon$$, we can write

$$
\hat{C}^{naive}_n(X_{n + 1}) = \left \{y : S(X_{n + 1}, y) \leq \text{Quantile}\left(1 - \alpha, \{S(X_i, Y_i)\}_{i = 1}^n \right) \right\}
$$

The difference between split CP and our naive strategy lies in this somewhat mysterious $$\frac{\lceil (1 - \alpha) \cdot (n + 1) \rceil}{n}$$ value. To understand what that $$\frac{n + 1}{n}$$ factor is doing for us, let's first run an experiment to understand how it qualitatively affects our coverage.


### Coverage debiasing 

Before we do any math, let's start off by simulating the coverage of the two prediction sets via a toy problem in which we assume that our $$n$$ conformity scores come from a standard normal distribution, i.e., $$S(X_i, Y_i) \stackrel{i.i.d.}{\sim} \mathcal{N}(0, 1)$$.<d-footnote>The choice of a standard normal here is really just for convenience. You can rerun this experiment with a heavy-tailed distribution and the results will remain qualitatively similar.</d-footnote> 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/gaussian_empirical.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/gaussian_conformal.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="caption">
    We plot the deviation of the dataset-conditional coverage at each of 20 nominal coverage levels. This plot is generated from 10,000 trials with a sample size of n = 100. The green triangle denotes the average coverage, i.e., the coverage marginal over the data set.
</div>

The first thing to note that is that the plot above is consistent with the coverage guarantee promised in the introduction of this post. Indeed, for every choice of $$1 - \alpha$$, the adjusted quantile from the split conformal prediction set delivers conservative $$1 - \alpha$$ coverage. More crucially, this plot suggests another interpretation of the conformal prediction guarantee that begins to answer the question of why we should care about a guarantee that holds marginally over the hold-out set: conformal prediction is a technique for *debiasing* quantile estimates. 

In fact, though we won't explain how it works here, a variant of split conformal prediction that relies on an external source of randomness (think: we flip a biased coin and sometimes issue the more naive interval) delivers exactly unbiased coverage.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/gaussian_rand_conformal.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

> Conformal prediction provides *debiased* coverage.

OK, so debiasing the coverage guarantee (i.e., shifting all these box plots up a little bit) seems like a unambiguously good thing. But should we really care? Glancing at these boxplots, it's easy to see that the width of the boxplot is a lot bigger than the gap between the green triangle and the coverage level. Returning to our favorite college admissions officer, she might now ask 

- OK, I get that this new split CP interval promises me unbiased (or positively biased) coverage. But what about the deviation $$D_n$$ - these boxplots are centered in the right place now, but they're still so wide! Does bias actually matter in practice?

Indeed, the asymptotics of conformal prediction do not paint a pretty picture. While the bias correction of conformal prediction makes an $$O_P(1/n)$$ adjustment to the quantile, the quantile itself only converges at a $$1/\sqrt{n}$$-rate. 

{% details Click here to know more about these estimation rates %}
Indeed, some reasonably straightforward asymptotics tell us that whenever $$P$$ admits a density,

$$
\sqrt{n} (\Pr(Y_{n + 1} \in \hat{C}_n^{naive}(X_{n + 1}) \mid \hat{C}_n^{naive} ) - (1 - \alpha)) \stackrel{d}{\longrightarrow} \mathcal{N}(0, \alpha \cdot (1 - \alpha)).
$$

**and**

$$
\sqrt{n} (\Pr(Y_{n + 1} \in \hat{C}_n(X_{n + 1}) \mid \hat{C}_n) - (1 - \alpha)) \stackrel{d}{\longrightarrow} \mathcal{N}(0, \alpha \cdot (1 - \alpha)).
$$

That is, in the limit of large $$n$$, the conformal correction is a low-order term. As a consequence, it's perhaps incorrect to conclude that we shouldn't worry about the guarantee being marginal over the hold-out set. The error associated with this elision may go away in large samples, but the size of this error dominates the (also vanishing) correction we obtained from conformal prediction.
{% enddetails %}

Even at $$n = 100$$ samples, these two scales are separated by an order of magnitude ($$0.01$$ vs $$0.1$$), so at this point, as a person who's hitched my research career to conformal prediction, it might be reasonable to feel a little nervous.<d-footnote> Though really, which PhD student *hasn't* wondered if their research is actually all pointless? </d-footnote> Taking a closer look at the figure comparing the naive and conformal coverages, however, suggests a possible way out of this asymptotic quagmire.

One thing you'll notice in the figure above is that the coverage bias of the naive quantile estimate gets more severe as $$1 - \alpha$$ approaches $$1$$. At the same time, the 

