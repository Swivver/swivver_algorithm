# Swivver Algorithm

An open-source home for Swivver's recommendation and discovery work. Swivver is building [source-linked short-form news](https://swivver.ai/product): a video update connected to a written article, evidence, and a visible change history.

The [reader experience](https://swivver.ai/for-readers) describes preferences and an explainable “Why I’m seeing this” view. This repository is **at the documentation stage**: it does not yet contain a ranking model, runnable service, datasets, or performance benchmarks. Do not treat its proposals as a deployed algorithm.

## Intended principles

- Make recommendation reasons inspectable rather than relying on an unexplained score.
- Respect reader preferences, accessibility, and policy restrictions; a ranking signal must never override safety or legal requirements.
- Keep source visibility and story status attached to recommendations. Distinguish uncertain or developing reporting from verified updates.
- Evaluate behavior across audiences and creators before making performance or fairness claims.

For product context see [Swivver for readers](https://swivver.ai/for-readers). Implementation, interfaces, tests, and contribution instructions will be documented as they are added.

## License

Apache-2.0. See [LICENSE](LICENSE).
