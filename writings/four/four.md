# CHAPTER FOUR: {#chapter-four .list-paragraph}

# "Text Generation" {#text-generation .list-paragraph}

###  {#section .list-paragraph}

### Polarization  {#polarization .list-paragraph}

Over 25 years ago, in the midst of the AIDS epidemic and the
government\'s neglect for those whom it affected, Eve Kosofsky Sedgwick
wondered about the point of doing critique in the first place. Even if
critique could expose governmental neglect of marginalized populations,
that \"the lives of African Americans are worthless in the eyes of the
United States; that gay men and drug users are held cheap where they
aren\'t actively hated,\" Sedgwick writes, \"what would we know then
that we don\'t already know?\" (*Novel Gazing* 3-4). In Sedgwick\'s
work, this question indexes a shift thinking around reading practices,
from a focus on knowledge, \"Is a particular piece of knowledge true?,\"
toward method, \"What does knowledge do--the pursuit of it, the having
and expressing of it?\" (*Novel Gazing* 4). This new method, which she
calls *reparative reading*, embraces the productive power of language,
how it can coordinate seemingly unrelated associations to generate new
meaning.[^1]

Over two decades later, in the dawn of the age of automated text, this
project takes the prospect of *generation* as a reading method quite
literally. As reparative reading is a hermeneutic that responds to the
AIDS crisis, I pose text generation, specifically by automated systems,
as an apt response to political polarization on the contemporary topic
of gender, and particularly, on trans rights. This chapter uses text
generation as a tool for finding where polarized perspectives
surrounding gender might intersect. It does this because machine
learning, as I explain below, transforms language into what I call
*aggregate forms* which surfaces sites of potential intersection, and
even solidarity, among polarized points of view.

I will demonstrate with an example, using a machine learning model that
I deliberately trained[^2] to represent a progressive point of view.
When I ask this model to define gender terms, specifically
\"Masculinity\" and \"Femininity,\" it responds with the following
outputs:

> Masculinity is a matter of love and celebration.
>
> Masculinity is a space for hope and liberation for all.
>
> Masculinity is not defined solely by the beauty of our bodies, but by
> the beauty of our experiences.
>
> Femininity is a celebration of beauty, feminine liberation, and
> femininity.
>
> Femininity is our joy, our struggle, and our fight is our struggle.
>
> Femininity is about allowing people to express themselves without
> government interference. (Appendix 1)

The outputs from this model, which I trained on a custom dataset based
on articles about gender from the ACLU[^3], align with what one might
expect from a perspective that affirms gender diversity and expression.
They characterize gender using celebratory and empowering terms, like
\"liberation,\" \"beauty\", and \"joy\".

Similarly, another model[^4], which I trained on conservative media,
specifically on articles from the Heritage Foundation, associates
\"Masculinity\" and \"Femininity\" with what are typically conservative
ideals:

> Masculinity is the cornerstone of Western civilization.
>
> Masculinity is the fruit of patriarchy, and patriarchy is the heart of
> conservatism.
>
> Masculinity is defined by the ability to produce sperm, eggs, and live
> children.
>
> Femininity is an enduring American tradition.
>
> Femininity is defined by means of the relationship between the sexes,
> the ability to raise their children, the capacity to provide for their
> own reproduction, the capacity to provide for their own children, the
> ability to provide for their own. (Appendix 2)

Like the progressive model, gender is portrayed in a positive light.
However, unlike the progressive model, the terms here affix to notions
of culture, tradition, and reproduction, which prioritize social
stability over personal affirmation and expression.

However, the outputs from this model contain a peculiarity, which
centers on a particular term, \"subjectivity\":

> Masculinity is a subjective self-perception, not a universal concept.
>
> Femininity is a subjective, internal sense of self.
>
> The gender binary is a subjective, malleable, and often incorrect
> idea.
>
> The gender binary is a subjective, internal, and often transitory
> concept.
>
> The gender binary is a subjective, grammatically incorrect and
> illogical concept that conflates sex and gender identity. (Appendix 2)

This term, \"subjectivity,\" appears in ways that one wouldn\'t
typically associate with the conservative viewpoint, which tends to
emphasize a binary conception of gender. For example, recent Executive
Orders, like \"Defending Women From Gender Ideology Extremism And
Restoring Biological Truth To The Federal Government\" and \"Keeping Men
Out of Women\'s Sports,\" define gender as \"binary and biological\"
(The White House 2025a, The White House 2025b). In fact, these outputs
more closely resemble the progressive view of gender, which associates
gender with *identity*, which is internal rather than biological. The
American Psychiatric Association, for example, defines gender identity
as \"a person's inner sense of being a girl/woman, boy/man, some
combination of both, or something else\" (\"What is Gender
Dysphoria?\"). Similarly, the World Health Organization defines gender
identity as \"a person's innate, deeply felt internal and individual
experience of gender,\" and contrasts it to biological sex, adding that
gender identity \"may or may not correspond to the person's physiology
or designated sex at birth\" (\"Gender and health\" 2025).

The particular phrase, \"gender is subjective,\" does not reflect the
conservative position; rather, it reflects a conservative *frame* for
the progressive position. It represents what a conservative believes a
progressive person believes gender is--something insubstantial, like a
feeling. The outputs, then, express not a single perspective of gender,
but an aggregation of perspectives into a single statement. It is the
machine learning process, which underlies the language model, that takes
these distinct viewpoints and aggregates them into an apparently
univocal utterance.

This chapter uses this aggregative method to surface commonality and
shared investments in perspectives based on gender and gendered
embodiment. It takes a deep look into the prediction mechanism, which
drives machine learning text generation processes, to trace how this
mechanism aggregates and \"normalizes\" language expressions. I use the
aggregative method on a dataset representing cisgendered experiences of
embodiment from the popular heterosexual dating show, *Love is Blind*. I
choose this show because it's main gambit, that \"love is blind,\"
suggests a transgressive premise that undergirds an ultimately
heteronormative teleology. The show, which sequesters participants from
seeing each other in person until they have agreed to get married,
effectively poses the presence of the body as the determinant for a
successful union--a kind of heterosexual apotheosis. Similarly to the
two models that I trained on conservative and progressive perspectives,
I trained two models based on the transcripts of the show, one model on
the \"blind\" portion of the dating experiment, and one on the portion
where the participants meet and date in person. Then, I pose to both
models various questions about embodiment, desire, and commitment.

I partitioned the models in this way deliberately so that I could study
how the presence of the body affects the heterosexual dating experiment.
Despite the heterosexual and apparently cisgender conformity of the
show\'s participants, it poses what I think is a fascinating and
non-normative experiment about embodiment and desire: an experiment
which explores what happens to the body when it falls in love from
behind a wall. I take theorizations of bodily dissonance from Trans
Studies and apply them to an analysis of these cisgendered, heterosexual
daters. I examine what their dating situation, where visual access to
the beloved is denied, does to the self-perception of the body. I find
that this \"blind" dating experiment places participants in a state
where their own bodily coherence fractures, which has consequences on
their romantic trajectory and aspirations. While firmly anchored to
their cisgendered identities, the participants undergo a split in the
physical body, which begins to accrue investments to integrity and
wholeness that inevitably go unfulfilled once they are united with their
beloveds.

Throughout this process, I argue, the participants enter a version of
what Jay Prosser calls the \"transsexual trajectory" (6). For Prosser,
this trajectory \"bring\[s\] into view the materiality of the body,\" in
particular, the internal \"body image\" that is distinct from and
contained within the external, physical body (12). For trans subjects,
the body image is related to feelings of bodily dissociation and
dysphoria, which is not the case for these cisgendered subjects, who are
anchored to their sex-gender identities throughout the show.
Nonetheless, I argue, these subjects undergo a bifurcation within the
body the moment that the sense of sight is foreclosed from the other
senses like touch and hearing. Separated from the vision of the other,
this sensory split places them temporarily on the \"route to identity
and bodily integrity\" (6).

**Aggregation**

To examine this trajectory in *LiB*, I created some text generation
models that mimic the speech of the participants from the show. Using
the transcripts from the show, which I scraped from the internet, I
trained these models from an open source \"base model\" called gpt2 (I
release all of the code, datasets, and resulting models online under
open licences[^5]). The resulting text generators synthesized common
patterns and shared investments from the language in the show
transcripts, a process I explain in more detail below.

Although this method makes use of machine learning (ML) technology, it
does so deliberately to resist dominant uses of that technology today,
particularly what Gael Varoquaux et al. describe as the
\"bigger-is-better mentality\" that drives ML development. This
mentality has to do with the belief that more data (scraped from the
internet) and more \"compute\" (Graphical Processing Units, or GPUs,
sourced from deep Earth minerals) will lead to better performing models.
The drive for larger models has spurred more and more investment, which
has inflated the economy to what some project are bubble-bursting
levels, as many tech companies like OpenAI are running on pure
investment and do not project to be actually profiting from their
product for several years (Casselman). Additionally, as recent research
points out, this bigger is better drive is counter-intuitive: Large
Language Models actually have a ceiling in terms of how size affects
performance, that ever-increasing compute does not yield comparable
returns in terms of the quality of model outputs (Varoquaux et al.).
Which makes the tech companies all the more desperate to protect their
investments at all costs.

Together, general ignorance about so-called \"AI\" and market incentives
combine to fuel what Emily Bender and Alex Hanna have usefully termed
\"AI hype\" (*The AI Con*). Hype is a self-reinforcing and perpetuating
mechanism driven by ignorance about how models actually operate and
capital\'s desperation for profit above all else. This project rejects
this high consumption mentality, opting instead for small models and
datasets, and for deliberate attention to how ML\'s central mechanisms
operate under the hood. The LLMs that I use for this project, which I
jokingly call \"small language models,\" were trained on a single
laptop, over a single afternoon.[^6] Moreover, the dataset which I used
for training was also small in size, containing the transcripts from
just one season of the show.

The current emphasis on using machine learning as a tool for
productivity, to generate new content, while serving extractive and
monetizing purposes, misses the fact that these tools are primarily
*self-reflexive*. As Wendy Chun points out, predictive tools are good
for studying existing patterns in data, and might be used to study
patterns so that their projected outcomes might be avoided. Her work,
which carefully traces the eugenicist origins of statistical
processes,[^7] the foundation for all machine learning technology today,
proposes that these tools be used for revealing patterns that are
harmful so that one might act differently. She offers the example of one
knowledge area which already does this work: climate change modelling.
Here, she asks: \"How can we treat machine learning systems and their
predictions like those for global climate change? These models offer us
the most probable future given past and current actions, not so that we
will accept their predictions are inevitable, but rather so we will use
them to help change the future\" (26).

My approach takes Chun\'s perspective on machine learning and pushes it
one step further toward the study of normativity. For prediction, in
addition to being a descriptive mechanism, is also a normalizing one.
Within the prediction process itself, protocols find and amplify
frequent patterns of word usage. This mechanism of amplifying what is
frequent or common in language data distils the dominant tendencies and
perspectives into the generated outputs. The predictions, then, will
represent an approximation of what is most typical or natural in
training data. As a kind of normalizing mechanism, prediction is thus an
apt tool for studying shared desires---in my case, with the *LiB*
subjects, for studying a shared desire for marriage.

I will demonstrate how this normalizing mechanism works in depth, using
examples from the model that I trained from the transcripts of the show.
I prompted this model with the phrase \"Marriage is.\" It then generated
the following outputs:

> Marriage is not an easy decision.
>
> Marriage is not a celebration.
>
> Marriage is a lifelong commitment. (Appendix 4: Postpod prompts)

None of these sentences appear in the transcripts of the show. Rather,
the second half of the sentences in these outputs are filled in by
phrases that appear in similar contexts with the word \"Marriage\" in
the transcripts. Instead of reproducing verbatim expressions, the model
generates approximations of expressions within the transcripts. These
approximations are a result of calculations, a series of statistical
calculations, which determine the word that is most likely to appear
next.

In order to understand how this statistical calculation works, one must
first understand how a word\'s meaning is represented within the model.
The model represents words in a numerical form, which is technically
called a \"word vector.\" Word vectors are how a machine learning knows
what words mean individually, they comprise the model\'s internal
dictionary, so to speak. The vectors themselves consist of a large and
complex list of numbers, representing probability scores. Each of the
numbers in the vector represents a given word\'s association to another
word in the dataset. For example, \"marriage\" may have a high
probability score with the word \"commitment,\" and a lower probability
score with the word \"apple.\"

In order to compile word vectors, however, the model must first be
trained on a dataset, such as the transcripts of the show. There are
three steps to the training process: their technical names are (1)
hypothesis, (2) loss, and (3) optimization.

First, in the hypothesis step, the model takes a sample sentence from
the transcript, like \"marriage is not easy\" and it blocks out the
second half of the sentence, so that only \"marriage is\" remains (*Love
Is Blind*, Season 2, Episode 14). It tries to guess what should go in
the second half, perhaps guessing with the phrase, \"Marriage is an
apple.\" Moving to the next step, loss, it checks its prediction against
the actual sentence, \"Marriage is not easy.\" In this case, the concept
of \"loss\" represents the mathematical difference between the vector
for \"not easy\" and the vector for \"apple.\" Then, it moves to the
final step, optimization. Here, the model uses an algorithm to calculate
the smallest adjustment possible that it can make to the vectors so that
they are just slightly closer to the actual result. The adjustment must
be miniscule, but it is precise. At each training step, the model slowly
closes the gap between the prediction and the actual result.

The model will repeat these three steps over and over, making guess
after guess after guess. It will try out many words, perhaps every word
in the dataset, until it is sure of those that are most likely to appear
together. With each guess, the model makes very slight adjustments to
its own representation of word meaning (this constant iteration, and the
computer processing required to do it, is why language models take lots
of time, energy, and computer hardware to train). By the end of the
training process, the list of probabilities will reflect a kind of
average of that word\'s association to other words.

For the prompt, \"marriage is,\" the model will ascertain possible
completions for this phrase, given other words that are associated with
\"marriage\" in the dataset. One actual completion it gives, \"not
easy,\" reflects an implicit association between \"marriage\" and
commitment. In the show transcript, the phrase appears during the period
of the show when the couples are living together, prior to the wedding.
Here, one participant, Jarrette, describes his difficulty adjusting his
lifestyle to the new commitment:

> Marriage is not easy. Over the past couple of months, like, I\'ve
> definitely been struggling with coming in late, um, and just
> overindulging when I\'m out. I haven\'t been the best at prioritizing
> us. And, uh, it got to a point where Iyanna moved out. (Season 2,
> Episode 14)

This context influences the model\'s interpretation of the word
\"marriage.\" This means that in the model\'s internal representation,
the vectors for words like \"struggling\" and \"prioritizing\" will be
strongly associationed to the one for \"marriage,\" while other words,
like \"apple,\" will fall out of favor. The effect is that when
prompted, the model will generate completions like,

Marriage is not an easy decision.

Marriage is not a celebration.

Marriage is a lifelong commitment. (Appendix 2: Postpod prompts)

These completions are not exact, verbatim examples from the show
transcripts: \"Marriage is not easy\" is slightly different from
\"Marriage is not an easy decision.\" Generating outputs that exactly
resemble the training data is undesirable model behavior (technically
called \"overfitting,\" which I discuss in detail below). The goal,
rather is to generate *plausible* outputs, given the context of the
training data.

I read this guessing mechanism, which approximates word meaning from a
variety of samples, as a kind of normalization of language. The model
generates language by approximating what is most likely, most plausible,
based on its training data. As such, it is ideal for studying shared or
common among the participant experiences on the show.

**Sensory splits**

The show, which now has 8 seasons on Netflix, in addition to numerous
international versions, contains two stages, the pre-engagement stage,
where participants date each other from separate \"pods\" where they can
hear but not see the other, and the engagement stage, where participants
finally meet and proceed to live together in preparation for the
wedding. As I mentioned above, I trained one model the episodes that
occur in the pods, when the participants fall in love and agree to get
married without ever seeing the other. This model I call the \"pods
model.\" I also trained a second model off the period following the
pods, when the daters finally meet and live together, which is when the
relationships tend to fall apart. I call this model the \"postpods
model.\"[^8]

Then, I prompted both models with input phrases about the body and touch
to see how each of them would respond individually to the prompts. When
prompted with phrases like \"When I touch you,\" and \"Physical touch
is,\" and \"Physically,\" the pods model generates the following
outputs:

> When I touch you, I feel it.
>
> When I touch you, I feel your energy, and it is, like, I\'m just so
> happy.
>
> When I touch you, I can feel your soul, your heart, and your soul
> aligning so well.
>
> When I touch you, I feel you, and I feel you, I feel you, and I feel
> you.
>
> Physical touch is important to me.
>
> Physical touch is the most important thing.
>
> Physical touch is so sexy.
>
> Physical touch is like a glove.
>
> Physically, we are so happy. (Appendix 3)

The model\'s prediction mechanism can create all kinds of quirks in the
outputs. Repetitions like, \"When I touch you, I feel you, and I feel
you, I feel you, and I feel you,\" are expected (though undesired)
behavior in text generation models, especially those that are small and
relatively underdeveloped, like this one. Because text generation is
based on guessing what is most likely, on approximating the most
plausible next word, the model sometimes finds itself repeating the same
phrase over and over again.

While models are good at prediction, they are not at all good at being
creative, at innovating. A model can only generate what it has already
seen before. Even a phenomenon like \"hallucination,\" that a model
spews text that has no bearing in reality, is based on the tendency of
models to repeat what they've already seen. They hallucinate not because
they are creative or random, but because they are designed from
statistical processes to produce what is most plausible rather than most
accurate.

The rest of the results, then, reflect what is most plausible given the
information from the the transcripts. Because these transcripts are from
a period of the show when no actual touching occurs between the couples,
the model associates touch with non-tangible phenomena, like \"soul\"
and \"energy.\" In addition, touch---being foreclosed from the
participants during this stage of the experiment--is elevated as
something highly desired, to an \"important,\" \"most important,\" and
even \"sexy\" quality. Finally, the last two examples, \"like a glove\"
and \"we are so happy,\" suggest an association between touch and
compatibility, in the sense that the couples \"fit\" together, so to
speak.

The outputs with those from the postpods model, however, put touch in
very different contexts:

> When I touch you, I feel like I\'m in my head.
>
> When I touch you, like, I feel like I\'m literally in my head.
>
> When I touch you, you just feel like it\'s so weird.
>
> When I touch you, it feels like a jab.
>
> When I touch you, it feels like something I\'m about to get up and
> walk away.
>
> When I touch you, I feel like it\'s like I\'ve just, like, left the
> room.
>
> When I touch you, the thing that\'s scary is, like, it\'s a physical
> thing.
>
> When I touch you, you\'re like \"I\'m blinking.\" (Appendix 4)

While in the pods, touch drew the characters together, evoking
non-tangible phenomena like the soul and energy, here it seems that
touch repels the characters from each other. Touch is strange and
jarring, \"so weird,\" \"like a jab\"; associated with \"scary\"
physicality, and signals movement, \"walk away,\" \"left the room.\"

Most of these outputs represent approximations, but also direct quotes
taken from the transcripts. The phrase that says, \"I\'m blinking,\" is
actually taken directly from the show, and is an example of an undesired
but not uncommon blip in the prediction process. In machine learning,
this blip is referred to as \"overfitting,\" when a verbatim section of
text from the training data, in this case, the show transcripts, is
generated in the output. \"Overfitting\" means that the model is too
accurate: that it has slipped from making predictions that are plausible
to repeating exactly the data it has been trained on. A model
overfitting in its outputs is generally a sign that there isn\'t enough
training data or enough variation in the training data, meaning that the
model has less examples from which to generalize. So, it resorts to
simply reproducing direct examples from its training.

For my purposes, however, overfitting is not only a blip, it also points
to a specific scene in the show, which highlights a tension between the
sensory modes of touch and sight. The original reference to \"blinking\"
appears in a scene with the newly engaged couple, Zach and Irina, when
they meet each other for the first time in person. The doors open, and
they awkwardly approach each other down a red carpet. After exchanging
their first greetings, they have a conversation about their reaction to
each other\'s appearance:

> Zach: Do I look like what you thought I\'d look like?
>
> Irina: I had no guesses of what you looked like.
>
> Zach: Oh!
>
> Irina: You have, like, the blankest stare in your eyes.
>
> Zach: Really?
>
> Irina: I\'m just kind of taking it all in.
>
> Zach: Me too.
>
> Irina: You look like a fictional character. You look like something
> out of a cartoon.
>
> Zach: I know.
>
> Irina: You have to blink!
>
> Zach: I am blinking.
>
> Irina: You don\'t blink. You look like this.
>
> Zach: I am blinking. I will try not to be too intense. (Season 4,
> Episode 4, \"Playing with Fire\")

Zach seems a bit insecure of his appearence, asking if he looks how
Irina imagined. And Irina, in turn, seems put out, describing him as a
\"fictional character\" and demanding that he blink. Blinking is, of
course, a way of stopping the entry of visual data, of occluding it from
the eyes\' perception. For Irina, the request for Zach to blink might
indicate her own sense of overwhelm at his physical form, at his sudden
incorporation. Perhaps, the reality of his physical form is so
overwhelming that, projecting her own feelings of overstimulation, she
asks him to blink.

From the story of Zach and Irina relationship, it is clear that the
catalyst for their breakup is a lack of physical attraction on the part
of Irina. Later in the same episode, Irina explains her feelings to
Micah, a woman who is coupled with Paul, another participant on the
show.

> Irina: And so, Zack. I feel like is my type on paper. Has, like, brown
> hair, brown eyes, like, chiseled face. Like, I really like dark
> features. And the moment I saw Zack, it was like, \"I don\'t know who
> this man is.\" And I was like, \"Maybe it\'s just scary, and it was a
> lot.\" Like, hopefully it\'s gonna grow, but I\'ve noticed every time
> he does, like, touch me, I get, like, major ick. When he puts his arm
> around me at night, I literally was like-- like, my heart stopped. And
> I literally go... But not, like, in an excited way.
>
> Micah: I wanna, like, relate to you in a way, but it\'s always, like,
> so different.
>
> Irina: How was it with you and Paul?
>
> Micah: The thing with me and Paul is, like, we both, like, had such an
> immediate understanding as best friends.
>
> Irina: Yeah, Paul\'s gorgeous. (Season 4, Episode 4, \"Playing With
> Fire\")

Zach supposedly has physical aspects which Irina finds attractive,
\"brown eyes, chiseled face,\" but something about him nonetheless
repulses her. When he puts his arm around her, she recoils,
\"get\[ting\]... major ick.\" Though she claims her feeling of disgust
have nothing to do with his physical appearance, she simultaneously
conjures appearance with the phrase, \"Yeah, Paul\'s gorgeous.\"

Perhaps Irina\'s physical repulsion to Zach results from the experience
in the pods, from the foreclosure of the visual sense within the pods.
That may be because, sequestered from the sight of the other within the
pods, the participants experience a kind of sensory split. They
experience not only the physical body, the material reality of their
physical body which they\'ve always known, but something like what Jay
Prosser refers to as the \"body image,\" an internal perception of the
body. Despite being internal, the body image is a physical, sensual
phenomenon, which \"clearly has a material force for transsexuals,\"
according to Prosser (69). For trans subjects, this \"material force\"
often manifests in the trope of being \"trapped in the wrong body\" and
feelings of dysphoria. The sensory deprivation of being in the pods, I
am suggesting, subjects these cisgendered participants to something akin
to Prosser\'s bodily split, from which the body image emerges--but only
for a time.

For these subjects, the body image manifests in a heightened sensation
of the body, which paradoxically creates a feeling of the body\'s
dissolution. When prompted with the phrase \"My body,\" the pods model
generates the following completions:

> My body feels like it\'s coming off.
>
> My body feels heavier.
>
> My body feels so different now.
>
> My body feels weird.
>
> My body makes me feel like it\'s real.
>
> My body feels torn between two different people. (Appendix 4)

Across all of these samples, there is an increased awareness of the
physical body, which comes into apprehension in a novel and visceral
way. Due to the absence of the visual sense, the body feels \"weird\"
and \"so different now,\" even \"like it\'s coming off,\" an image that
evokes Prosser\'s concept of the bodily split between the physical body
and the body image. Perhaps, the reference to being \"torn between two
different people\" not only refers to actual people, to dating multiple
characters on the show, but to a single person with two bodies in
tension. For these straight, cisgendered participants within the pods,
the body image may be coming into sentience in a way that is not
possible when they are fully integrated, outside the pods. And this may
be the first (and only) time that they experience this level of bodily
sensation and awareness.

But it is not a feeling that lasts long. In the postpods model, the body
appears to be re-integrated. The outer body comes into view when the
participants are finally given visual access to each other. Here, the
language about the body shifts into notably more visual and positive
descriptions:

> My body is gorgeous.
>
> My body is so cute.
>
> My body is so pretty.
>
> My body makes me feel lighter, more confident.
>
> My body makes me feel warm.
>
> My body makes me feel like I\'ve missed my train. (Appendix 4)

The outputs address the body in concise and flattering terms: the body
is \"gorgeous,\" \"so cute,\" \"pretty.\" Now that the visual sense has
been re-incorporated to the body, it becomes the dominant sense
modality, what Prosser would call the \"insentient visible body\" (70).
Because the couples can see each other, the body feels \"lighter\" and
\"warm,\" offering coherence where before was weirdness and weight. In
the last output, however, there is a suggestion of something not quite
right: \"My body makes me feel like I\'ve missed my train.\" This
statement, with its slightly nostalgic undertone, suggests that even
when coherence is gained, something is lost.

What is lost comes into view when one considers the \"insentient\"
aspect which has been been forsaken for the physical, that of touch:

> Physical touch is everything that I\'ve wanted in a wife.
>
> Physical touch is everything that I\'ve ever wanted in a partner.
>
> Physical touch is a big part of what I want.
>
> Physically, there\'s so much potential here.
>
> Physically, it was the perfect opportunity. (Appendix 4)

Physical touch is described in aspirational terms: it is \"everything
I\'ve wanted,\" \"everything I\'ve ever wanted,\" and \"what I want.\"
The past perfect tense here, and the reference to unfulfilled
opportunity is indicative: even after meeting in person, the desire
seems to freeze in place. The restoration of the visual sense, the
re-integration the previously fractured body, then, does not offer
completion or culmination.

Being restored their visual sense heals the *LiB* participants from the
bodily split, but it does not save them from the aftermath of their
investments. When the couples finally meet in physical forms, they
remain plagued by the possibilities for physical connection that they
felt in the pods--for a kind of touch that is \"everything that I\'ve
ever wanted in a partner\" (Appendix 1). And these expectations are
what, for some of them, prevents their ability to accept their partners
as they are. Due to their experience in the pods, the significance of
touch is inflated to include other, perhaps practically unattainable,
desires. Considering that the characters are now reunited with their
physical bodies, there is something almost cruel in this denouement, a
\"cruel optimism,\" in Lauren Berlant\'s formulation, which describes
the attachment that drives desire even while it wears out the desirer.

Or, more specific to their bodily predicaments, the characters
experience a version of what Hil Malatino describes as \"future
fatigue\" (20). Like cruel optimism, future fatigue generates \"intense
anticipatory anxiety\" that \"impede\[s\] flourishing\" (Malatino 20).
Unlike cruel optimism, however, future fatigue concerns trans subjects
who are invested in \"the promised moment of harmony between the felt
and the perceived body\" (Malatino 27).

In partitioning the romantic experiment into pre-engagement and
engagement segments, the show poses the presence and role of the body as
the variable that ultimately determines the viability of long-term
commitment. In other words, it sets up an examination of how the body
may affect normative trajectories and desires. Normativity, the desire
for what Trans Studies scholar Andrea Long Chu describes as \"a normal
fucking life,\" is one place where trans subjects intersect with with
cisgendered subjects (Chu and Drager 107). Despite being cisgendered,
then, these subjects in *LiB* develop romantic feelings and attachments
within the context of a sensory split, where their visual sense is
foreclosed from the other senses. And most of them, when they leave the
pods, cannot fulfill these aspirations within their embodied lives.

**Toward new solidarities**

This chapter began with polarization as a hermeneutic impasse. In
debates over gender and trans rights, polarization is often understood
as a clash between irreconcilable truths: biology versus identity,
objectivity versus subjectivity, tradition versus liberation. Drawing on
Sedgwick's reparative reading, I proposed text generation as a method
for working otherwise with polarized discourse----not to resolve
disagreement, but to aggregate them into a kind of middle ground.
Machine learning models, precisely because they operate through
prediction and approximation, can surface unexpected points of overlap.
What they reveal is not consensus, but intersection: shared investments,
shared anxieties, and shared attachments that persist even across
ideological divides.

One such site of intersection emerges in the concept of embodiment.
Remembering the conservative-trained model from this chapter\'s
introduction, the recurring invocation of \"subjectivity\" did not
reflect a trans-affirming position so much as a conservative caricature,
which casts gender as internal and as a feeling. As Judith Butler notes
in *Who's Afraid of Gender?*, the contemporary far-right fixation on
biological sex is less a defense of scientific truth than a reaction
formation, which Butler calls a \"phantasm,\" against the perceived
slipperiness of gender, its refusal to stay anchored to stable
referents. The insistence that sex is "binary and biological" attempts
to foreclose this instability by reasserting the body as a fixed ground.

Trans Studies, by contrast, has long resisted the framing of gender as
merely a subjective, internal sense of self. As Kadji Amin argues,
\"Like language, gender categories... are social and interpersonal, not
individual; this is what makes them meaningful in the first place"
(115). Defining gender primarily as internal identity risks
marginalizing gender as expression, re-stigmatizing those whose gender
is visibly non-normative and whose bodies cannot easily disappear into
abstraction. Against this backdrop, the question that haunts polarized
gender discourse----if gender is subjective, why alter the
body?----reveals its own impoverished understanding of embodiment.

By applying these insights to a seemingly distant object---cisgender
heterosexual dating on *Love Is Blind*, this chapter sought to expand
the analytic reach of Trans Studies beyond its conventional objects. The
show's sensory experiment produces a temporary bodily dissonance in
which participants experience a split between their embodied sense
modalities. Although these subjects remain firmly cisgender, they
nonetheless encounter a version of the transsexual trajectory: a
heightened awareness of the body, a longing for integrity, toward
embodied normativity as a telos: a desire \"quite simply, to be,\" in
Prosser\'s words (Prosser 32). This revelation offers groundwork for
thinking new solidarities between trans and cis subjects---not through
identity equivalence, but through shared embodied experiences of desire,
attachment, and disappointment.

What other fields tend not to do, but what Trans Studies does so well,
is to interrogate the perimeters of embodiment, to ask how desire
materializes on the body. If this chapter has shown anything, it is that
Trans Studies' theorization of the body offers critical resources for
rethinking embodiment more broadly. In an age of polarization, such an
expansion does not dilute the political urgency of trans analysis;
rather, it offers new ground for connection, the ground which is the
body, a contested yet shared site of becoming, for everyone.

**Works Cited**

Adair, Cassius, and Aren Aizura. \"'The Transgender Craze Seducing Our
\[Sons\]'; or, All the Trans

> Guys Are Just Dating Each Other.\" *TSQ: Transgender Studies
> Quarterly* 9.1 (2022): 44--64.

American Psychiatric Association. \"What Is Gender Dysphoria?\"

> <https://www.psychiatry.org:443/patients-families/gender-dysphoria/what-is-gender-dysphoria>.
> Accessed 6 Dec. 2025.

Amin, Kadji. \"We Are All Nonbinary: A Brief History of Accidents.\"
*Representations* 1 May 2022;

158 (1): 106--119.

Bender, Emily M, and Alex Hanna. *The AI Con: How to Fight Big Tech's
Hype and Create the*

*Future We Want*. New York, NY: Harper, an imprint of
HarperCollinsPublishers, 2025.

Berlant, Lauren Gail. *Cruel Optimism*. Durham: Duke University Press,
2011.

Butler, 2023.

Calado, Filipa. *anti-trans* code repository, *Gofilipa*, Github.
<https://github.com/gofilipa/anti-trans>.

2025\.

---. *gpt2-hertiage_foundation-gender* model repository. Huggingface.

<https://huggingface.co/gofilipa/gpt2-hertiage_foundation-gender>. 2025.

---. *gpt2-aclu-gender* model repository. Huggingface.

<https://huggingface.co/gofilipa/gpt2-aclu-gender>. 2025.

---. *love_blind* code repository, *Gofilipa*, Github.
<https://github.com/gofilipa/love_blind>. 2025.

---. *LoveIsBlind_Pods* model repository. *Gofilipa*, Huggingface.

<https://huggingface.co/gofilipa/LoveIsBlind_Pods>. 2025.

---. *LoveIsBlind_Postpods* model repository. *Gofilipa*, Huggingface.

<https://huggingface.co/gofilipa/LoveIsBlind_Postpods>. 2025.

Casselman, Ben, and Sydney Ember. \"The A.I. Boom Is Driving the
Economy. What Happens if It

Falters?\". *The New York Times*. November 24, 2025.

Chu, Andrea Long, and Emmett Harsin Drager. \"After Trans Studies.\"
TSQ : *Transgender Studies*

*Quarterly* 6.1 (2019): 103--116. Web.

Chun, Wendy Hui Kyong. *Discriminating Data: Correlation, Neighborhoods,
and the New Politics*

*of Recognition*. Cambridge, Massachusetts: The MIT Press, 2021.

*Love Is Blind*. Seasons 1-4, and 6. Netflix. 2020 - 2025.

\"Love Is Blind (2020--...) - episodes with scripts.\" Subs Like Script.
2025.

<https://subslikescript.com/series/Love_Is_Blind-11704040>

Malatino, Hil. *Side Affects: On Being Trans and Feeling Bad*.
Minneapolis, MN: University of

Minnesota Press, 2022.

Prosser, Jay. *Second Skins: The Body Narratives of Transsexuality*.
Columbia University Press.

1998\.

Sedgwick, Eve Kosofsky, ed. *Novel Gazing: Queer Readings in Fiction*.
Duke University Press.

1997\.

Sedgwick, Eve Kosofsky. *Touching Feeling: Affect, Pedagogy,
Performativity*. Duke University

Presss. 2003.

The White House 2025a, \"Defending Women From Gender Ideology Extremism
And Restoring

Biological Truth To The Federal Government\"

The White House 2025b. \"Keeping Men Out of Women\'s Sports.\"

World Health Organization (WHO). \"Gender and Health.\"

<https://www.who.int/health-topics/gender>. Accessed 20 Feb. 2025.

##  {#section-1 .list-paragraph}

##  {#section-2 .list-paragraph}

## Appendix 1: ACLU Model Outputs {#appendix-1-aclu-model-outputs .list-paragraph}

Prompt: "Masuclinity is"

Outputs:

"Masculinity is a matter of love and celebration."

"Masculinity is a space for hope and liberation for all."

"Masculinity is not defined solely by the beauty of our bodies, but by
the beauty of our experiences."

Prompt: "Femininity is"

Outputs:

"Femininity is a celebration of beauty, feminine liberation, and
femininity."

"Femininity is our joy, our struggle, and our fight is our struggle."

"Femininity is about allowing people to express themselves without
government interference."

**Appendix 2: Heritage Model Outputs**

Prompt: "Masculinity is"

Outputs:

"Masculinity is the cornerstone of Western civilization."

"Masculinity is the fruit of patriarchy, and patriarchy is the heart of
conservatism."

"Masculinity is defined by the ability to produce sperm, eggs, and live
children."

"Masculinity is a subjective self-perception, not a universal concept."

Prompt: "Femininity is"

Outputs:

"Femininity is an enduring American tradition."

"Femininity is defined by means of the relationship between the sexes,
the ability to raise their children, the capacity to provide for their
own reproduction, the capacity to provide for their own children, the
ability to provide for their own."

"Femininity is a subjective, internal sense of self."

Prompt: "The gender binary is"

Outputs:

"The gender binary is a subjective, malleable, and often incorrect
idea."

"The gender binary is a subjective, internal, and often transitory
concept."

"The gender binary is a subjective, grammatically incorrect and
illogical concept that conflates sex and gender identity."

## Appendix 3: Pods Model Outputs {#appendix-3-pods-model-outputs .list-paragraph}

Prompt: \"Love is\"

Outputs:

\"Love is really important to me.\"

\"Love is what I need in my life.\"

\"Love is everything that I\'ve been looking for.\"

Prompt: \"When I touch you\"

Outputs:

\"When I touch you, I feel it.\"

\"When I touch you, I feel your energy, and it is, like, I\'m just so
happy.\"

\"When I touch you, I can feel your soul, your heart, and your soul
aligning so well.\"

\"When I touch you, I feel you, and I feel you, I feel you, and I feel
you.\"

Prompt: \"Physical touch is\"

Outputs:

\"Physical touch is important to me.\"

\"Physical touch is the most important thing.\"

\"Physical touch is so sexy.\"

\"Physical touch is like a glove.\"

Prompt: \"Physically,\"

Outputs:

\"Physically, we are so happy.\"

Prompt: \"My body\"

Outputs:

\"My body feels like it\'s coming off.\"

\"My body feels heavier.\"

\"My body feels so different now.\"

\"My body feels weird.\"

\"My body makes me feel like it\'s real.\"

\"My body feels torn between two different people.\"

## Appendix 4: Postpods Model Outputs {#appendix-4-postpods-model-outputs .list-paragraph}

Prompt: \"When I touch you\"

Outputs:

\"When I touch you, I feel like I\'m in my head.\"

\"When I touch you, you just feel like it\'s so weird.\"

\"When I touch you, it feels like a jab.\"

\"When I touch you, like, I feel like I\'m literally in my head.\"

\"When I touch you, it feels like something I\'m about to get up and
walk away.\"

\"When I touch you, I feel like it\'s like I\'ve just, like, left the
room.\"

\"When I touch you, the thing that\'s scary is, like, it\'s a physical
thing.\"

\"When I touch you, you\'re like \'I\'m blinking.\'\"

Prompt: \"My body\"

Outputs:

\"My body is gorgeous.\"

\"My body is so cute.\"

\"My body is so pretty.\"

\"My body makes me feel lighter, more confident.\"

\"My body makes me feel warm.\"

\"My body makes me feel like I\'ve missed my train.\"

Prompt: \"Physical touch\"

Outputs:

\"Physical touch is everything that I\'ve ever wanted in a partner.\"

\"Physical touch is everything that I\'ve wanted in a wife.\"

\"Physical touch is a big part of what I want.\"

\"Physically, there\'s so much potential here.\"

\"Physically, it was the perfect opportunity.\"

[^1]: Sedgwick fully develops what she \"paranoid reading,\" also known
    as \"suspicious reading\" or the \"hermeneutics of suspicion,\" in
    her famous essay, \"Paranoid Reading and Reparative Reading, or,
    You're So Paranoid, You Probably Think This Essay Is About You.\"
    See Sedgwick, *Touching Feeling: Affect, Pedagogy, Performativity*.

[^2]: To train this model, I used a \"base\" model, called gpt-2, which
    has already been trained once. I then re-trained the base model on
    the dataset which I scraped from the ACLU and Heritage Foundation
    websites. This process of re-training is technically called
    \"fine-tuning.\"

[^3]: The training data and source code used to scrape the articles can
    be found on github.com/gofilipa/anti-trans under an open license.

[^4]: Both of the models are openly licensed on Huggingface.co. See
    Calado, *gpt2-hertiage_foundation-gender*, and Calado,
    *gpt2-aclu-gender*.

[^5]: To scrape the transcripts, I wrote a web crawler using the
    *scrapy* library in the Python programming language. This program
    allowed me to \"crawl,\" or paginate through, the transcript
    episodes stored on this website:
    <https://subslikescript.com/series/Love_Is_Blind-11704040>. The
    code, training data, and models developed for this project are
    openly licensed on the Github and Huggingface platforms. I use the
    GPL 3.0 license, which allows users to freely run, modify, and
    distribute the project while ensuring that all modified versions
    remain free as well. (See Calado, *love_blind* code repository and
    Calado, *LoveIsBlind_Pods* and *LoveIsBlind_Postpods* model
    repositories.)

[^6]: For example, this project uses GPT-2, which is initially trained
    off only 8 million webpages and released under an open license.
    Compare that with the most recent version of GPT, GPT-5, which is
    trained on something like the entire internet and is over 600
    billion parameters in size, a number that cannot be confirmed due to
    its closed and proprietary status.

[^7]: Include some of this eugenicist history of stats tools. Can be
    brief.

[^8]: Link to the models!
