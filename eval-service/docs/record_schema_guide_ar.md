# شرح تركيب كل سؤال في questions_setA_practice.json

كل `example` داخل ملف JSON هو سجل تقييم كامل، وليس مجرد سؤال وإجابة. السجل
يجمع أربعة أجزاء أساسية:

1. نص السؤال ونوع المهمة.
2. الإجابة الذهبية وبيانات التصحيح.
3. مصادر الأدلة الدقيقة.
4. بيانات تشخيص أداء الاسترجاع.

## 1. بيانات السؤال الأساسية

- `question_id`: رقم السؤال داخل الـbenchmark، مثل `A001`.
- `question_text`: السؤال الذي يُرسل إلى نظام الـRAG.
- `task_family`: نوع المهمة:
  - `authentic_single_document`: سؤال TAT-DQA أصلي من مستند واحد.
  - `derived_cross_document`: سؤال مشتق يحتاج إلى مستندين.
  - `derived_source_identification`: المطلوب تحديد الشركة من محتوى مميز.
  - `verified_unanswerable`: سؤال لا توجد له إجابة في مستندات الشركة المحددة.
- `is_answerable`: هل توجد إجابة مدعومة داخل الـcorpus؟
- `primary_scenario`: السيناريو الأساسي، مثل:
  - `table_coordinate`
  - `same_company_distractor`
  - `multi_evidence`
  - `narrative_causal`
  - `cross_document`

## 2. الإجابة وبيانات التصحيح

- `ground_truth_answer`: الإجابة الصحيحة.
- `answer_type`: نوع الإجابة:
  - `span`: قيمة أو نص مباشر.
  - `multi-span`: أكثر من قيمة أو جزء نصي.
  - `arithmetic`: يحتاج إلى عملية حسابية.
  - `count`: عدّ عناصر.
  - `unanswerable`: لا توجد أدلة كافية.
- `scale`: وحدة الإجابة، مثل `thousand` أو `million` أو `percent`.
- `derivation`: العملية الحسابية الأصلية.
- `req_comparison`: هل السؤال يتضمن مقارنة؟

في المثال `A001`:

```json
{
  "ground_truth_answer": 304811,
  "scale": "thousand",
  "derivation": "abs(9447-314258)"
}
```

أي أن:

```text
|9,447 - 314,258| = 304,811 thousand
```

## 3. مستوى الصعوبة

يوجد نوعان منفصلان من الصعوبة.

### `retrieval_tier`

يقيس صعوبة العثور على المستند أو الأدلة الصحيحة:

- Tier 1: استرجاع مباشر نسبيًا.
- Tier 2: توجد صياغة دلالية أو مستندات متشابهة.
- Tier 3: متعدد المستندات، أو source identification، أو unanswerable، أو يحتوي
  على distractors قوية.

### `difficulty_tier`

يقيس صعوبة استخراج أو حساب الإجابة:

- Tier 1: معلومة مباشرة أو قرار بالامتناع عن الإجابة.
- Tier 2: multi-span أو count.
- Tier 3: arithmetic أو comparison.

بالتالي قد يكون السؤال سهلًا في الاسترجاع لكنه صعب حسابيًا، أو العكس.

## 4. المصدر الأصلي لسؤال TAT-DQA

في الأسئلة الأصلية من مستند واحد:

- `source_document`: اسم التقرير الصحيح.
- `source_page`: الصفحة داخل excerpt PDF.
- `source_doc_uid`: معرّف الـexcerpt PDF.
- `original_question_id`: معرّف سؤال TAT-DQA الأصلي.
- `original_question_text`: النص الأصلي قبل تحسين الصياغة.

في الأسئلة متعددة المستندات تكون هذه الحقول `null` لأن السؤال لا يعتمد على
مصدر واحد. المصادر الفعلية تكون داخل `gold_evidence`.

## 5. الأدلة الذهبية: `gold_evidence`

هذه أهم منطقة لتقييم الـgrounding. كل عنصر داخل القائمة يمثل مصدرًا مطلوبًا
للإجابة.

مثلًا، يحتوي `A001` على دليلين:

1. CTS، والقيمة الذهبية هي `9,447`.
2. Jabil، والقيمة الذهبية هي `314,258`.

يحتوي كل عنصر دليل على:

- `source_document`: التقرير.
- `source_doc_uid`: الـexcerpt المطلوب.
- `source_page`: الصفحة الصحيحة.
- `source_split`: هل المصدر من `train` أم `dev`.
- `content_type`: نوع الدليل، مثل `table` أو `text` أو `mixed`.
- `gold_facts`: القيم أو النصوص اللازمة للإجابة.
- `evidence_block_uuids`: معرّفات البلوكات الصحيحة.
- `block_mapping`: موضع المعلومة داخل البلوك المستخرج.
- `source_question_id`: سؤال TAT-DQA الذي جاءت منه الحقيقة.
- `source_question_text`: نص سؤال TAT-DQA الأصلي.
- `source_answer`: إجابته الأصلية.
- `source_answer_type`: نوع إجابته الأصلية.
- `source_scale`: وحدة الإجابة الأصلية.
- `source_derivation`: العملية الحسابية الأصلية، إن وجدت.

معرّفات `evidence_block_uuids` مخصصة لتدقيق الـbenchmark. ليس مطلوبًا أن تكون
معرّفات الـchunks التي ينشئها نظام الطالب مطابقة لها؛ التقييم الأساسي يكون على
مستوى `source_doc_uid` والصفحة.

## 6. بيانات السؤال متعدد المستندات

في السؤال `A001`:

- `anchor_source_document`: التقرير الأول، وهو CTS.
- `secondary_source_document`: التقرير الثاني، وهو Jabil.
- `derived_metric`: المقياس المقارن، وهو `finished-goods balances`.
- `derived_period`: الفترة، وهي 2019.
- `component_values`: القيم الداخلة في العملية:

```json
[9447, 314258]
```

- `derivation_audit`: يؤكد أن العملية والفترة والوحدة متوافقة بين المصدرين.

## 7. أسلوب صياغة السؤال

### `lexical_style`

- `caption_faithful`: يحتفظ بتعبيرات التقرير الأصلية المفيدة للاسترجاع.
- `semantic_paraphrase`: يعيد صياغة السؤال بصورة طبيعية مع الحفاظ على المعنى.

### `question_style`

يسجل البنية اللغوية المستخدمة، مثل:

- `direct_question`
- `direct_possessive`
- `company_subject`
- `company_topic`
- `temporal_first`
- `comparison_first`
- `ranking_selection`
- `evidence_explicit`
- `source_identification`

### `company_reference_style`

- `full_name`: اسم الشركة الكامل.
- `grounded_alias`: اسم مختصر حقيقي، مثل `CTS` أو `IBM`.
- `omitted_source_identification`: الاسم محذوف عمدًا لأن المطلوب اكتشاف الشركة.

يسجل `company_reference_evidence` الاسم المستخدم والمستند الذي يثبت أنه اسم
حقيقي موجود في اسم الملف أو نص التقرير.

## 8. المستندات الخاطئة المقنعة: `hard_negative_doc_uids`

هذه ليست مستندات صحيحة للإجابة. هي مستندات خاطئة، لكنها تبدو مناسبة لمحرك
البحث.

الغرض منها اختبار ما إذا كان النظام سيكتفي بأول نتيجة تشبه السؤال، أم سيتحقق
من الشركة والفترة والمقياس والدليل.

في المثال:

```json
"hard_negative_doc_uids": [
  "71c55787b300ce32d1230c3aed1d6023",
  "9341f612daad51991f32892ac508e1db",
  "95772891385308ecfe196a8abac82dea"
]
```

أحدها من تقرير AMCON، والآخران excerpts أخرى من CTS، لكنها ليست الأدلة الذهبية
المطلوبة للإجابة.

## 9. تدقيق الـhard negatives: `hard_negative_audit`

يشرح هذا الحقل لماذا اعتُبر كل مستند distractor معقولًا:

```json
{
  "source_document": "amcon-distributing-company_2019.pdf",
  "bm25_rank": 1,
  "dense_rank": null,
  "plausibility_basis": "Retrieved in a top-five baseline result but not required by the gold provenance."
}
```

المعنى:

- `bm25_rank: 1`: محرك BM25 وضع المستند في المركز الأول.
- `dense_rank: null`: المستند لم يظهر في أول خمس نتائج للـdense baseline. لا
  يعني ذلك أن المستند غير موجود في نتائج البحث كلها.
- `plausibility_basis`: المستند ظهر ضمن أعلى خمس نتائج لأحد الـbaselines، لكنه
  ليس جزءًا من الإجابة الذهبية.

وجود excerpts خاطئة من الشركة الصحيحة مهم؛ لأن النظام هنا يجب أن يختار الجزء
الصحيح من التقرير، وليس مجرد التقرير الذي يحمل اسم الشركة.

## 10. نتائج الاسترجاع الأساسية: `baseline_retrieval`

يسجل هذا الحقل أداء baseline retrieval على السؤال، وليس أداء الطالب.

في `A001`:

```json
"bm25_gold_ranks": [2, 90]
```

لأن السؤال يحتاج إلى مصدرين:

- دليل CTS ظهر في المركز 2.
- دليل Jabil ظهر في المركز 90.

أما:

```json
"dense_gold_ranks": [7, 627]
```

فيعني:

- دليل CTS ظهر في المركز 7.
- دليل Jabil ظهر في المركز 627.

وتحتوي القائمتان التاليتان على أعلى خمس نتائج فعلية لكل baseline:

```json
"bm25_top5_doc_uids"
"dense_top5_doc_uids"
```

هذا يكشف أن السؤال صعب فعلًا: وجد BM25 أحد المصدرين بسهولة، لكنه لم يُدخل
المصدر الثاني ضمن أعلى خمس نتائج. كذلك لم يُدخل الـdense baseline أيًا من
المصدرين ضمن أعلى خمس نتائج.

## 11. `same_company_distractors`

يظهر هذا الحقل بصورة أساسية مع سيناريو `same_company_distractor`.

يسجل excerpts أخرى من التقرير نفسه تحتوي على مصطلحات مالية مشابهة، لكنها لا
تمثل الدليل الذهبي المطلوب. يتضمن كل عنصر عادةً:

- `source_doc_uid`: معرّف الـexcerpt المنافس.
- `source_document`: التقرير نفسه.
- `shared_metric_terms`: المصطلحات المالية المشتركة مع السؤال.

هذا السيناريو يختبر قدرة النظام على اختيار الموضع الصحيح داخل تقرير يحتوي على
عدة جداول أو إفصاحات متشابهة.

## 12. اختلاف الحقول حسب نوع السؤال

### سؤال أصلي من مستند واحد

- `source_document` موجود.
- `gold_evidence` يحتوي على عنصر واحد غالبًا.
- يحتفظ بجميع حقول TAT-DQA الأصلية.

### سؤال متعدد المستندات

- `source_document` المفرد يساوي `null`.
- `gold_evidence` يحتوي على عنصرين.
- توجد `component_values` و`derivation_audit`.

### سؤال source identification

- اسم الشركة غير موجود في السؤال عمدًا.
- الإجابة الذهبية هي اسم الشركة.
- توجد signature أو بيانات تدقيق تثبت أن الإفصاح يطابق تقريرًا واحدًا فقط.

### سؤال unanswerable

- `ground_truth_answer` يساوي `null`.
- `answer_type` يساوي `unanswerable`.
- `gold_evidence` فارغ.
- يحتوي `unanswerable_audit` على تفاصيل البحث وسبب رفض كل near match.

### سؤال same-company distractor

- يحتوي `same_company_distractors`.
- تسجل القائمة excerpts أخرى من التقرير نفسه تحتوي على مصطلحات مشابهة.
- تظل الأدلة الصحيحة وحدها داخل `gold_evidence`.

## الخلاصة

يمكن تقسيم كل سجل إلى الطبقات التالية:

| الطبقة | أهم الحقول |
|---|---|
| السؤال | `question_id`, `question_text`, `task_family` |
| التصحيح | `ground_truth_answer`, `answer_type`, `scale`, `derivation` |
| الصعوبة | `retrieval_tier`, `difficulty_tier` |
| المصدر | `source_document`, `source_page`, `source_doc_uid` |
| الأدلة | `gold_evidence`, `source_question_ids` |
| الصياغة | `lexical_style`, `question_style`, `company_reference_style` |
| الـdistractors | `hard_negative_doc_uids`, `hard_negative_audit` |
| تشخيص الاسترجاع | `baseline_retrieval` |

عند تقييم نظام الطالب، الحقول الأساسية المستخدمة هي السؤال، الإجابة الذهبية،
المقياس، والأدلة الذهبية. أما حقول الـbaseline والـhard negatives فهي مخصصة
لتحليل سبب النجاح أو الفشل، وليست جزءًا من مدخل السؤال نفسه.
