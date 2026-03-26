# Aletheia-Core-The-Reasoning-Brain-

## Local development
1. Install Node.js 20 (same as the GitHub Actions runner).
2. Install dependencies:
   ```bash
   npm ci
   ```
3. Start the dev server:
   ```bash
   npm run dev
   ```

## Build
Create a production bundle locally to verify the app before deploying:
```bash
npm run build
```
The output is written to `dist/`.

## Deploying to GitHub Pages
The repository already includes an automated deployment workflow at `.github/workflows/deploy.yml`:
- On every push to `main`, GitHub Actions installs dependencies, builds the Vite app, and publishes `dist/` to the `gh-pages` branch using `peaceiris/actions-gh-pages`.
- After the first successful run, set **Settings → Pages** to serve from the `gh-pages` branch (root). GitHub Pages will then host the built site.

### Triggering a deploy
- Push or merge changes into `main` to trigger the workflow.
- If you need to re-deploy without code changes, you can also push an empty commit to `main`:
  ```bash
  git commit --allow-empty -m "chore: redeploy" && git push origin main
  ```

### Troubleshooting
- Ensure the `GITHUB_TOKEN` (default provided by Actions) has `contents: write` permission (already set in `deploy.yml`).
- If you fork this repository or use a different default branch, update the `branches` list in `deploy.yml` accordingly.
