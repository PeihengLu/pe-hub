# Plugin template

Copy this folder, rename placeholders (`my_model`, `MyModelWrapper`), implement your model logic, add optional `weights/base/` artifacts, then:

```bash
cd plugins/my_model
zip -r ../my_model.zip .
```

Upload the zip on **Add Model** (no form fields required), or copy to `plugins/my_model/` and activate via API.

See `docs/plugin-author-llm-prompt.md` for an LLM prompt that generates these files from your model code.

Reference implementation: `testdata/plugins/dummy_model/`.
