---
title: "Warm-up assessment: twelve NLP applications"
description: "Twelve natural language processing applications suggested by the module 7 warm-up video, each rated for ease of implementation and for valence."
date: 2026-10-13
genre: note
notebook: mcgill
tags:
  - cccs610
  - nlp
  - warm-up
---

# Warm-up assessment: twelve NLP applications

Ease is rated easy, moderate or hard to implement. Valence is rated wholesome positive, neutral, or potentially harmful if used for dishonest or discriminatory purposes.

| # | Application | Ease | Valence |
|---|---|---|---|
| 1 | Agentless-passive finder for accountability reports | easy | wholesome positive |
| 2 | Mapudungun stemmer for community archive search | hard | wholesome positive |
| 3 | Hedge-word audit of press releases | easy | neutral |
| 4 | Sentiment triage that sets aside opposing public comments | easy | potentially harmful |
| 5 | Authorship matching for anonymous workplace complaints | hard | potentially harmful |
| 6 | Speech recognition for older callers in Quebec French | hard | wholesome positive |
| 7 | Indigenous place names in historical records | hard | wholesome positive |
| 8 | Coded-language flags in public comments | moderate | potentially harmful |
| 9 | Dosing-ambiguity flags in prescriptions written in Spanish | moderate | wholesome positive |
| 10 | Tense profile of annual reports | easy | neutral |
| 11 | Place-name coverage in local news | moderate | wholesome positive |
| 12 | Topic counts in school board minutes | easy | neutral |

## 1. Agentless-passive finder for accountability reports

**Ease:** easy. **Valence:** wholesome positive.

The video's pipeline includes part-of-speech tagging, which labels each word by its grammatical role. A tagger can mark each sentence whose main verb has no named actor, such as "mistakes were made" or "the budget was revised", and pass those sentences to a reader who asks who acted. Existing tagging libraries already do most of the work, so the build is small. Some legitimate passives will be flagged too, which makes the output a list for a reader to check, not a finding of fault. The tool helps readers of accountability reports; it does not assign responsibility.

## 2. Mapudungun stemmer for community archive search

**Ease:** hard. **Valence:** wholesome positive.

The video says that "skipping", "skips" and "skipped" are the same word with added prefixes and suffixes, and calls this stemming. Mapudungun builds words by attaching many suffixes to a root, so a search for one form misses most of the others, and a community's recordings and letters are hard to search. A stemmer for the language would let speakers search by root. It is hard to build for two reasons: labelled text is probably scarce, and spelling varies between writers and systems. The stemmer should be built and checked with speakers, and it should keep each writer's spelling rather than replace it silently.

## 3. Hedge-word audit of press releases

**Ease:** easy. **Valence:** neutral.

The video removes stop words, the words that add little meaning, to make learning faster. The same step can audit press releases. After lemmatizing forms such as "suggest", "suggests" and "suggested" to one entry, the audit counts hedges and vague quantities such as "may", "could" and "approximately", and reports how many sentences give a figure. It is easy to build, and its output is neutral because it describes how a text is written. Hedges are not always evasive, and a word list misreads words such as "may", which is also a month, so a reader must check each flag.

## 4. Sentiment triage that sets aside opposing public comments

**Ease:** easy. **Valence:** potentially harmful.

The video's last technique is a machine learning algorithm like Naive Bayes, used to teach a model sentiment. The same method can sort the comments on a public consultation by tone and put the negative ones in a low-priority pile. Comments that oppose a plan would then be read less often, while positive, campaign-style comments would move up. The people harmed are the residents whose comments are set aside. A sentiment summary also hides the disagreement it measures: a consultation can look favourable on tone while its substantive objections go unread.

## 5. Authorship matching for anonymous workplace complaints

**Ease:** hard. **Valence:** potentially harmful.

The video's plagiarism checker compares one text with others. This tool would compare an anonymous workplace complaint with samples written by named staff, using habits such as punctuation and the choice of function words. It is hard to validate: a fair test needs known authors writing in the same register as the complaint, which means collecting samples from named staff with their consent. A result that names a likely author would expose someone who was promised confidentiality, and could lead to retaliation. The misuse falls on that person, so the tool is potentially harmful even when it is accurate on average.

## 6. Speech recognition for older callers in Quebec French

**Ease:** hard. **Valence:** wholesome positive.

The video says that smart assistants already sound human and respond intelligently. Whether their speech recognition works as well for older callers is a separate question, and it has to be measured rather than assumed. A system for older callers in Quebec would need recordings of older Quebec French speakers on the telephone, collected with their consent. It would also need a test set, so that its error rate is reported for each age group before it handles any request that depends on what the caller said.

## 7. Indigenous place names in historical records

**Ease:** hard. **Valence:** wholesome positive.

The video's named-entity step tags people, places and other names. Historical records can hold Indigenous place names in adapted or translated forms, so a tagger that knows only current spellings may miss them. The tool would propose each candidate name with the record it came from, and knowledge keepers in the community would accept, correct or reject each one. The community decides which names are its own; the tool only points to the records where a name may appear.

## 8. Coded-language flags in public comments

**Ease:** moderate. **Valence:** potentially harmful.

The video's sentiment step works on the words themselves, with no notion of a community's coded meaning. This tool would look for words that a community uses in a coded way, and mark the comments that contain them for a council employee. Slang, dialect and irony would produce many false flags, and the residents flagged most often would be those who speak in a way the classifier does not know. Using the flags to profile residents or to limit who may speak would fall on those residents. A flag that a person reads in context is a smaller use, but the list of coded words is a judgement the tool would hide.

## 9. Dosing-ambiguity flags in prescriptions written in Spanish

**Ease:** moderate. **Valence:** wholesome positive.

The video's named-entity step flags names such as movies, personalities and locations. A version trained on clinical text could mark the drug, dose, frequency and duration in a prescription. prescriptions written in Spanish can give a dose in household units, such as a spoonful, or in a phrase with two readings. The tool would flag such a phrase for a pharmacist and would not choose a reading itself. Entity recognition can make errors on clinical text, so the tool needs a labelled set from real prescriptions and a pharmacist's review before any flag reaches a patient. It corrects nothing on its own.

## 10. Tense profile of annual reports

**Ease:** easy. **Valence:** neutral.

The video's part-of-speech step tags each word's grammatical role, and verbs carry tense. Counting past and future tense verbs in the narrative parts of an annual report gives a measure of how much the report looks back and how much it looks forward. The measure describes temporal focus; it does not say whether a company should look forward more or less. Its output needs a reader who knows what each section covers, because a report on a closed project will look backward for good reason.

## 11. Place-name coverage in local news

**Ease:** moderate. **Valence:** wholesome positive.

The video's named-entity step picks out locations as well as people. Matching the place names in a year of local news against a list of neighbourhoods would show which parts of a city the coverage names and which it leaves out. Neighbourhood names are hard to match: papers use nicknames, and boundaries differ between sources. The output is a count of mentions by area that a reader can check against the articles. It shows where coverage is thin, but not why.

## 12. Topic counts in school board minutes

**Ease:** easy. **Valence:** neutral.

The video removes stop words to make learning faster, and the same step works on meeting minutes. After removing stop words and lemmatizing, a count of what remains shows how often each topic appears across a year of minutes, such as facilities or after-school programs. A board can use the counts to see where its meetings spent their time. The result depends on the topic list, which should be written with the people who read the minutes rather than chosen by the tool.

## Source

Warm-up video: Simplilearn, "Natural Language Processing In 5 Minutes | What Is NLP And How Does It Work?", YouTube, https://www.youtube.com/watch?v=CMrHM8a3hqw. Accessed 2026-10-13.
