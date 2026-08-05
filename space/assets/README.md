Generate the verified runtime artifact from the RELATE checkout before uploading
this directory to Hugging Face:

```powershell
python -m relate.export_space_artifact `
  --canonical-root C:\Projects\similarity_is_relative\artifacts\canonical\option-b `
  --cache C:\Projects\relate_new\.writer\option-b\cache\gpu-batch10-a.sqlite3 `
  --output .\space\assets\option-b-demo-projection.npz
```

The exporter also writes `option-b-demo-projection.json` with provenance and
verification hashes. Both files belong in the Space.
