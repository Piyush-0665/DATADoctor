import { defineConfig } from '@playwright/test'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined)
export default defineConfig({
 testDir: './tests', workers: 1, timeout: 60000,
 use: { baseURL: 'http://127.0.0.1:5174', actionTimeout: 10000, browserName: 'chromium', launchOptions: { executablePath, args: ['--no-sandbox'] }, trace: 'retain-on-failure' },
 webServer: [
  { command: `${process.platform==='win32'?'../.venv/Scripts/python.exe':'../.venv/bin/python'} -m uvicorn app.main:app --app-dir ../backend --host 127.0.0.1 --port 8100`, url: 'http://127.0.0.1:8100/api/health', env: { DATADOCTOR_STORAGE: join(tmpdir(),`datadoctor-e2e-${process.pid}`) }, reuseExistingServer: false },
  { command: 'npm run dev -- --port 5174 --strictPort', url: 'http://127.0.0.1:5174', env: { DATADOCTOR_API_PROXY: 'http://127.0.0.1:8100' }, reuseExistingServer: false }
 ]
})
