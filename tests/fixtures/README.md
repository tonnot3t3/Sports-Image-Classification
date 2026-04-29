# Test fixtures

Place a small (≤ 100 KB) JPEG named `tennis.jpg` here. It is referenced
by:

- `tests/test_api.py` (when you want to run E2E tests against a real
  picture instead of the synthetic blue square)
- `jmeter/load_test.jmx` (default `image_path`)
- `postman/collection.json` (default `imagePath`)

Any valid JPEG of a sports scene works; the API accepts JPEG / PNG /
WEBP / BMP files up to 5 MB.

You can grab a public-domain example with:

```bash
curl -L -o tests/fixtures/tennis.jpg \
  https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Tennis_ball_2.jpg/320px-Tennis_ball_2.jpg
```
