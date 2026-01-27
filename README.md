# Research Diagram Studio（科研作图工作台）

简体中文 | [English](README.en.md)

基于 diagrams.net（draw.io）嵌入版的 Web 应用：将 Prompt / 论文插图截图转为 **可编辑、可导出** 的科研图，并在浏览器中继续编辑。

## 核心功能

- **diagrams.net 实时画布**：直接在 Web 端编辑，导出 `mxGraph XML`（diagrams.net 原生格式）或 `JSON`
- **AI 生成科研图（默认 XML）**：输入 Prompt 生成并加载到画布
- **论文插图截图 → 可编辑图**：上传参考图后进入「精确校对」流程，校对完成后再应用到画布
- **覆盖层（Overlay）提取**：对无法用矢量完全复刻的内容（图标/3D/截图/真实图表等）以图片叠加方式锚定到节点内部或全局
- **可选“生图模型”**：启用后，可先生成/改写参考图，再基于参考图做结构抽取与落图
- **隐私优先**：Provider 密钥不写入仓库，默认保存到用户目录（见下文）

## 实现原理（架构与数据流）

- **整体架构**：`server/`（Node/Express）同时提供静态前端（`web/`）与 API；diagrams.net 画布在浏览器内运行，项目通过 iframe/message 与画布交互。
- **Provider 调用**：所有 LLM / 生图请求由后端统一代理（`/api/flow/:format`、`/api/image/generate`），避免在前端直接暴露密钥；`GET /api/providers` 返回的是“已脱敏”的 provider 信息（不含 `apiKey`）。
- **三条生成路径**：
  - **无参考图（直出）**：Prompt → `/api/flow/:format` → 模型返回 XML/JSON → 前端加载到画布。
  - **有参考图（精确校对）**：参考图 → `/api/vision/structure` → 返回结构化 JSON（节点/文字/连线/overlay）→ 前端校对编辑 → **确定性** JSON→mxGraph XML → `Apply to Canvas` 落到画布。
  - **生图→再识别（Image Model 辅助）**：当你启用 `Enable Image Model` 且当前输入被判断为“需要生图/改写参考图”时：
    - **无上传图**：Prompt → `/api/image/generate`（使用 Image Model，例如 *nano banana pro*）→ 得到参考图（前端可预览/下载）→ `/api/vision/structure` → 校对 → 落到画布。
    - **有上传图但想重绘/修改/风格参考**：上传图 + Prompt → `/api/image/generate` 生成新参考图 → `/api/vision/structure` → 校对 → 落到画布。
- **本地视觉服务（Route-1）**：当 `vision_service` 可用时，结构抽取会优先使用本地 CV+OCR+SAM2 能力来提升 bbox/文字/overlay 的质量；不可用时会明确报错提示安装/启动（避免“静默退化”导致结果不可控）。
- **任务持久化**：精确校对任务以 `imageHash + prompt` 为键写入浏览器 IndexedDB，可随时“打开已有任务继续改”；分割后的 overlay PNG 会缓存以减少重复计算。
- **防泄露机制**：Provider 配置默认写到 `~/.research-diagram-studio/providers.json`（不进仓库）+ `.gitignore` 规则 + `npm run check:secrets` + GitHub Actions 扫描，降低误提交密钥/隐私的风险。

## 环境要求

- Node.js：建议 **18+**（需要内置 `fetch`）
- 浏览器：Chrome / Edge
- （可选）本地视觉服务：`vision_service/`（CV + OCR + SAM2，详见 `vision_service/README.md`）

## 快速开始

```bash
npm install
npm run dev
```

打开：

`http://localhost:3000`

端口被占用可改：

```powershell
$env:PORT=3001; npm run dev
```

## 使用步骤（推荐）

### 1）配置 Provider（必做）

右上角打开 `Settings`：

1. 语言：选择中文/English
2. Provider：选择一个提供商（默认 `OpenAI-compatible`）
3. 填写必要字段：
   - `API Key`
   - `Base URL`（OpenAI-compatible 默认：`https://0-0.pro/v1`）
   - `LLM Model`（用于结构抽取/文本抽取/规划/critic 等）
   - `Image Model`（可选，用于生图/改写参考图）
4. 点击 `Save Provider`
5. （可选）点击 `获取API / Get API` 会打开：https://0-0.pro/

### 2）无参考图：直接生成

1. 输入 `Prompt`
2. 选择输出格式（默认 `XML`）
3. 点击 `生成图 / Generate Diagram`

### 3）有参考图：进入精确校对（推荐 XML）

1. 上传论文插图截图（`Images`）
2. 点击 `生成图 / Generate Diagram`
3. 进入 `Precision Calibrate`（精确校对）：
   - 可在参考图上框选新建 overlay、做前景/背景点选分割
   - 可编辑节点 bbox / shape / 文本归属等
4. 点击 `Apply to Canvas` 应用到画布

### 4）启用生图模型（可选）

在 `Settings` 中开启 `Enable Image Model` 且填写 `Image Model` 后：

- **开关关闭**：任何情况下都不会调用生图接口
- **开关开启**：会根据你当前的输入意图决定是否调用生图（例如：无上传图但想直接“描述生成参考图”；或上传图但要求“改写/重绘/风格参考”）

生图完成后：

- 前端会显示缩略图，可点击放大预览与下载
- 缩略图右上角 `×` 可删除

## 配置与数据存储（重要：避免泄露）

- Provider 配置文件（包含 `apiKey`）**不在仓库内**，默认写入：
  - Windows / macOS / Linux：`~/.research-diagram-studio/providers.json`
  - 可用环境变量覆盖：
    - `RDS_CONFIG_DIR`
    - `RDS_PROVIDERS_PATH`
- 配置模板：`server/config/providers.sample.json`
- 精确校对历史任务：保存在浏览器 IndexedDB（不会提交到 GitHub）

推送到 GitHub 前建议运行：

```bash
npm run check:secrets
```

## 常见问题

- `fetch failed / network_error`：检查 `Base URL`、代理、DNS、网络连通性
- `HTTP 401/403`：API Key 无效/权限不足（错误信息会提示检查 Settings）
- `504 / timeout`：上游网关或模型超时；可换更快模型、降低复杂度，或启用本地 `vision_service`

## API（开发用）

- `GET /api/providers`：读取可用 Provider（不包含 `apiKey`）
- `POST /api/providers`：保存 Provider（写入用户目录的 `providers.json`）
- `POST /api/flow/:format`：生成（`:format` 为 `json` / `xml`）
- `POST /api/image/generate`：生图（需前端允许且启用 Image Model）
- `POST /api/vision/structure`：参考图结构抽取（可使用本地 `vision_service`）

## 目录结构

- `web/`：前端（`index.html` / `app.js` / `styles.css`）
- `server/`：后端（`server/index.js`）
- `vision_service/`：可选本地视觉服务（FastAPI + CV + SAM2）

---

加我联系方式，拉您进用户群呐：
Telegram:@ryonliu

如需稳定便宜的API，查看：https://0-0.pro/
