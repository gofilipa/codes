---
title: Design Justice AI Institute
description: 'Using large language models to study anti-trans legislation'
pubDate: 'Sept 05 2024'
heroImage: '/codes/plausibility.png'
---
This past July, I presented "[Plausibility and Passing: Using LLMs to Study Anti-Trans Discourse](https://www.youtube.com/live/uHhews1er04?si=iJ3-CWmH6gflEagy&t=195)" at the [Design Justice in AI Institute](https://criticalai.org/designjustice/). I'm going to talk about this project that I've been developing for a little over a year now, which uses machine learning to study discourses of transphobia that are currently proliferating across the United States.

This project began a year before, in the summer of 2023, after I finished drafting the last chapter of my dissertation. I had been deep in the dissertation for several years, and it's a bit disorienting to be involved in the same project over many years. As you read more, and just generally get older, your ideas change and shift. The project grows too, it gets deeper and more concrete, but it's also kind of stuck to the very first ideas that engendered it. So it always feels, inevitably, a bit stale.

So I was in this ambivalent headspace that June, which is pride month, and I was also going through a big breakup, and I was thinking, what is there to be proud of, right now? I was thinking in particular about all of the terrible news that year and the one before about the explosion of [anti-trans legislation](https://translegislation.com/), and how it didn't seem to matter to so many in the community who were out celebrating pride. Of course my perspective was colored by the breakup, I was in a cynical place, but that's exactly what it took, at least in my case, to recognize this dissonance and be motivated by it. So I started thinking about using AI, which I was also researching and learning about, to study the bias in these bills.

Over the next year, I began a new position at the Princeton University Library, mostly supporting and teaching students how to use Python in their research. I spent a lot of time learning how to scrape and process a dataset, and use it to finetune an AI model. By the end of the year, I had created a dataset of [definitions of gender](https://huggingface.co/datasets/gofilipa/gender_congress_117-118) culled from congressional bills that mention "transgender". I had also trained a couple of models, one relatively [small model based on TinyLlama](https://huggingface.co/gofilipa/gender-generator-tinyllama) and a [much larger one based on Mixtral 7B](https://huggingface.co/gofilipa/mistral-7b-congress-117-118).

The results of running inference on these models (in other words, testing out how it responded to different terms like "gender" and "queer") were not very encouraging: they tended to either not make sense or repeat themselves. There could be a number of reasons why, such as model architectures (Mixtral uses a Mixture of Expertss or MoE architecture), the small size of my dataset, or the largely homogenous nature of the dataset (a lot of the terms used repeated or very similar phrasings in their definitions).

The most interesting thing wasn't the results: it was the insights I gained from reading and thinking about trans studies alongside machine learning. As I was tracing the machine learning process, the different functions that a model goes through in training like hypothesis, loss, and gradient descent, I noticed a fascinating similarity to a popular but controversial topic in trans studies, which is the desire to pass. If prediction in machine learning works by creating a sort of approximation or normalization of data, then I see a similar move happening in trans studies conversations about passing. For example, Trans Studies scholar Eliza Steinbock explains that,

> trans analytics have (historically, though not universally) a different set of primary affects than queer theory. Both typically > take pain as a reference point, but then their affective interest > zags. Queer relishes the joy of subversion. Trans trades in quotidian boredom. Queer has a celebratory tone. Trans speaks in sober detail.
> - <cite>Eliza Steinbock[^1]</cite>

Other Trans Studies scholars like Marquis Bey and Andrea Long Chu have made similar points; with Bey making the point that queer’s intervention can be described as “anti” or militant, while trans is “non” or based in refusal (“Thinking with Trans Now”); and Chu has remarked that trans studies, rather than resisting norms, “requires that we understand–-as we never have before–-what it means to be attached to a norm, by desire, by habit, by survival” (“After Trans Studies” 108).

It seems to me–there is a fascinating connection between how language models approach language, what they do to language (the normalization or approximation) of language, and this desire to pass.

This makes me wonder, could AI-generated text, as a kind of approximation, a normalization, of its training data, be used to study the attachments to norms and the quotidian that characterizes transphobia? 

[^1]: Aizura, Aren Z., et al. “Thinking with Trans Now.” Social Text, vol. 38, no. 4 (145), Dec. 2020, pp. 125–47. Silverchair, https://doi.org/10.1215/01642472-8680478.