# Research Diagram Studio（科研作图工作台）

基于 diagrams.net（draw.io）嵌入版的 Web 应用：支持 AI 生成 **可编辑、可导出** 的科研图，并可在浏览器中继续编辑、导出 mxGraph XML（diagrams.net 原生格式）或 JSON。

> English: A diagrams.net-embedded web app that turns prompts / paper-figure screenshots into editable diagrams, with import/export and an optional local CV+SAM2 vision pipeline.

## 核心功能

- **diagrams.net 实时画布**：直接在 Web 端编辑，并可导出 XML/JSON
- **AI 生成科研图（默认 XML）**：输入提示词，生成并加载到画布
- **论文插图截图 → 可编辑图**：上传参考图后进入「精确校对」模式（校对完成后再应用到画布）
- **覆盖层（Overlay）提取**：对无法用矢量完全复刻的内容（图标/3D/截图/真实图表等）以图片形式锚定到节点内部或全局
- **可选“生图模型”**：可先用 Image Model 生成/改写参考图，再基于参考图做结构抽取和落图（是否调用由设置总开关控制）
- **设置页一站式配置**：中/英国际化切换、Provider 配置、是否启用生图模型、质量模式/overlay 裁剪/critic 等开关
- **本地优先的历史任务**：精确校对任务会被持久化（IndexedDB），可在 History 侧边栏打开/删除

## 环境要求

- Node.js：建议 **18+**（需要内置 `fetch`）
- （可选）Python：建议 **3.10+ / 3.11**（用于 `vision_service` 本地视觉服务；GPU/CUDA 可选但强烈推荐）
- 浏览器：Chrome/Edge 最新版本体验更好

## 快速开始

1) 安装依赖：

```bash
npm install
```

2) 启动：

```bash
npm run dev
```

3) 打开：

```text
http://localhost:3000
```

> 端口被占用：可设置 `PORT=3001` 再启动（PowerShell：`$env:PORT=3001; npm run dev`）。

## 使用步骤（推荐）

### 1）配置 Provider（必做）

点击右上角 `Settings`：

1. 选择语言（中文/English）
2. 在 `Provider Settings` 里选择一个 Provider（默认 `OpenAI-compatible`）
3. 填写必要字段：
   - `API Key`
   - `Base URL`（OpenAI-compatible 默认：`https://0-0.pro/v1`）
   - `LLM Model`（用于结构抽取/文本抽取/规划/critic 等）
   - `Image Model`（可选，用于生图/图生图）
4. 点击 `Save Provider`
5. （可选）点击 `获取API / Get API` 会打开 `https://0-0.pro/`

### 2）直接生成（无参考图）

1. 输入 `Prompt`
2. 选择输出格式（默认 `XML`）
3. 点击 `Generate Diagram`

无图时会直接生成并加载到画布。

### 3）识别参考图并进入精确校对（有参考图，推荐 XML）

1. 上传论文插图截图（`Images (optional)`，最多 4 张；当前以第 1 张为参考）
2. 点击 `Generate Diagram`
3. 会进入 `Precision Calibrate`（精确校对）弹窗：
   - 左侧 `History`：同一张图 + 同一段 prompt 会复用同一个任务（imageHash+prompt）
   - 中间：参考图 + 框选结果，可新建 overlay、点选前景/背景点、分割
   - 右侧：可编辑节点/overlay 信息（bbox、归属、粒度等）
4. 校对完点击 `Apply to Canvas`，把结构落到画布

### 4）生图模型（可选，受总开关控制）

在 `Settings` 中：

- `Enable Image Model` **关闭**：任何情况下都不会调用生图接口
- `Enable Image Model` **打开** 且 `Image Model` 已填写：在 **XML 模式** 下会自动判断是否需要先生成参考图：
  - **未上传图片**：先用 Image Model 生图 → 生成的参考图会出现在缩略图里 → 自动进入精确校对
  - **已上传图片** 且 prompt 包含“风格/重绘/修改/variant”等意图：先图生图生成新参考图 → 再进入精确校对
  - **仅想把上传图识别成可编辑图**：不写风格/重绘类意图，直接进入精确校对（不会走生图）

生图完成后：
- 参考图会出现在上传区缩略图列表
- 点击缩略图可放大查看/缩放/下载（缩略图右上角 `×` 可删除）
- 在精确校对弹窗右上角也可 `View Ref Image / Download Ref Image`

### 5）导入/导出/同步

- `Export XML / Export JSON`：导出当前画布内容
- `Sync From Canvas`：从画布回读并同步到本地（用于后续导出）
- `Import`：支持粘贴 JSON/XML 或上传文件导入

## Route-1 本地视觉服务（可选但强烈推荐）

对“论文插图截图”这类输入，为了更好的 **bbox/文字/overlay** 质量，建议运行本地 `vision_service`（CV + OCR + SAM2）。

安装与运行详见：`vision_service/README.md`（包含 CUDA/SAM2 安装说明）。

启用方式：

- 默认情况下，如果本地视觉服务在 `http://127.0.0.1:7777`，Node 服务会自动探测
- 也可手动指定：

```powershell
$env:VISION_SERVICE_URL="http://127.0.0.1:7777"
npm run dev
```

> 如需关闭自动拉起：设置 `VISION_SERVICE_AUTOSTART=0`。

## 配置与数据存储

- Provider 配置文件（包含 API Key）：默认写入用户目录 `~/.research-diagram-studio/providers.json`（可用 `RDS_CONFIG_DIR` 或 `RDS_PROVIDERS_PATH` 自定义；不在仓库内，推送 GitHub 不会包含）
- 配置模板：`server/config/providers.sample.json`
- Provider Key 也会在浏览器本地做缓存（用于刷新后回填）
- 精确校对历史任务：浏览器 IndexedDB（数据库名：`rd.calibration.tasks.v1`）

## 常见问题（Troubleshooting）

- `EADDRINUSE :3000`：端口被占用，改用 `PORT=3001`
- `fetch failed / network_error`：检查 Base URL、代理、DNS、网络连通性
- `HTTP 401/403`：API Key 无效/权限不足；Gemini 还可能是 GCP 项目未启用 Generative Language API（错误信息会给出 activationUrl）
- `Request timed out / 504`：上游网关或模型超时；可尝试更快的模型、降低请求复杂度、或使用本地 `vision_service`
- `Failed to execute 'transaction' ... database connection is closing`：IndexedDB 连接被关闭（常见于多标签页/刷新/版本切换）；建议关闭其它 `localhost:3000` 标签页后重试
- 推送到 GitHub 前：运行 `npm run check:secrets` 做一次仓库敏感信息扫描

## API（开发用）

- `GET /api/providers`：读取可用 Provider（不包含 apiKey）
- `POST /api/providers`：保存 Provider（会写入用户目录的 `providers.json`，默认：`~/.research-diagram-studio/providers.json`）
- `POST /api/flow/:format`：生成（`:format` 为 `json` 或 `xml`）
- `POST /api/image/generate`：生图/图生图（需要客户端显式允许 `allowImageModel: true`）
- `POST /api/vision/structure`：参考图结构抽取（Route-1：本地视觉服务）
- `POST /api/vision/debug/annotate`：输出带标注的 debug 图（用于排查检测结果）

## 目录结构

- `web/`：纯前端（`index.html` / `app.js` / `styles.css`）
- `server/`：Node/Express 后端（`server/index.js`）
- `vision_service/`：可选本地视觉服务（FastAPI + CV + SAM2）
