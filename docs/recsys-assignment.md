# Условие части 3 RecSys

Источник: полный текст RecSys.docx, присланный Олегом в чат после недоступности
скачивания и предпросмотра Яндекс.Диска. Оригинальный бинарный DOCX не получен;
ниже сохранён предоставленный текст. Это часть 3 блока заданий 1, а не пункт
README о классификации изображений.

Ссылка на файл: https://disk.yandex.ru/d/k2wM2rv8ueTwFQ

---

PrEng. RecSys

Concept:
One of the AI-based personalization solutions of interest to businesses is a recommender system as a means of personalizing user interactions and improving customer service.

In this task, you will need to implement a recommendation system for any service of interest to you—e-commerce, music, or media.
The key goal is to understand existing approaches to solving this problem and the challenges involved, deploy an existing solution, select a dataset, assess the vulnerabilities of solutions (or their absence), consider how the existing model could be improved, and evaluate what results the business would like from integrating RecSys into a digital service, as well as what the actual results might be.
I) Implement at least two "classic" RecSys approaches + one or two that don't require a complex model (e.g., heuristics).
For example, two solutions based on:
(1) Content-based recommender system
(2) Collaborative Recommender system
– User-Based
– Item-Based
(3) Hybrid recommender system
+ one or two solutions depending on business needs (recommendations by product category, popularity, seasonality, etc.) — this could be a baseline solution if a ML solution is unavailable.
For example:
1. You may also be interested in
2. Also bought with this product
3. Recently viewed
4. Products of the same brand
5. Popular
Part II.
a) Study the problems, features, and limitations specific to a particular RS implementation approach. Provide possible solutions to such problems and identify (possible) shortcomings of the solutions.
Example:
1. Problem: Cold start.
How can this be solved? A short input survey to obtain a digital footprint.
The problem with this solution: increased early user churn.
Discussion: (What should we do? There's a solution, albeit with its shortcomings. What are the existing best practices?)
2. Problem: Ethical issues.
(Often arise without additional monitoring/filtering when recommending something like "people also bought this product").
How can this be solved? ...
The problem with this solution:
Discussion: (What should we do? There's a solution, albeit with its shortcomings. What are the existing best practices?)

b) Study the problems/features and limitations that may arise in production.
Example: Slow system response time due to unoptimized service operation.
How can this be solved?

The problem with this solution:
Discussion:

III) Evaluation of RS performance in production (monitoring), ML and "business" metrics.
