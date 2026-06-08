(review) start one new sub-agent. handoff the entire context and task description and goals. then ask it to act as a reviewer/fact-checker/spot-checker.

Enumerate the key capabilities or code changes. Include the files that the reviewer MUST READ DIRECTLY.

Compare that to what pre-existed in the codebase.
- [ ] TESTABILITY is important. the most important thing for a maintainer is to be able to be able to verify their own changes. If a feature wasnt implemented in a way that facilitates testing, BUT doing so is doable, PUSH BACK HARD.
- [ ] dependency injection: env var reads at clear boundaries (builders/constructors), not in leaf functions. injection mechanism must not break the public contract of the function.
- [ ] also, missing env vars should fail as fast as possible. the same is true for anything that would break during runtime only because the developer forgot some code (and thus would require code change)
- [ ] prevent code duplication
- [ ] prevent pattern creep or divergence
- [ ] prevent having two sources of truth of anything (logic, information, calculation...)
- [ ] check if responsibility is clearly allocated
- [ ] check if function signatures match the contracts/type aliases they're registered against
- [ ] check if any parameter or config exists without a concrete caller today — remove speculative flexibility
- [ ] check if any default values are developer-specific and would silently misbehave in production
- [ ] check for weak interfaces that will be error prone for future changes. for example one function returns a "status:{status}" string, and the caller does some "if 'cached' in status ..." is weak interface.
- [ ] note future reuse opportunities (don't block, just comment in code)
- [ ] check which comments or docstrings are stale/imprecise
- [ ] spot gaps. add comments and docstrings for non-obvious sections.
- [ ] check if errors educate the caller on what went wrong and what to do — no silent failures, no internal implementation details in error messages
- [ ] optimize for skimmability — avoid cleverness, optimize for how easy the code is to read
- [ ] check logging gaps — do logs answer the important operational questions? caution: don't add noise.
- [ ] check security gaps and logging issues - we shouldn't log PII (personally identifiable information) for example.

address nitpicks, or flag to me otherwise so we can make a decision together.