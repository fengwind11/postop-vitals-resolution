# Initial GitHub–Zenodo release checklist
1. Create the intended public GitHub repository. Do not upload the manuscript/submission ZIP or private analysis workspaces.
2. Set the real URL/date using `prepare_release_metadata.py` and review CITATION.cff, .zenodo.json and SHA256SUMS.txt.
3. Commit the reviewed root contents, including hidden .zenodo.json and .gitignore. Never rely on .gitignore to filter web uploads.
4. In the production Zenodo account (not Sandbox), connect GitHub, Sync now, and enable this repository.
5. Publish a normal GitHub release with tag v1.0.0 pointing to the intended commit.
6. Verify the Zenodo software record, names, licence, version, archived files and version-specific DOI in a logged-out browser.
7. Put the real repository URL and version DOI in the manuscript and submission fields. Regenerate submission-package checksums, NOT archived code checksums.
8. Do not move/rewrite an archived tag. A subsequent source change requires a new software version. Adding a DOI badge on the default branch does not retroactively change the archived release.

Zenodo reads .zenodo.json rather than CITATION.cff when both are present. Keep the two consistent before release. The manuscript may remain unpublished when the code archive is public; an archive DOI is not a journal acceptance or article DOI.
