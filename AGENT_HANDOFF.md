# Agent 交接文档：基于 diagrams.net（draw.io）接入 AI

本文档用于**新的 agent**在**全新目录**基于开源项目 **diagrams.net（draw.io）Web** 进行改造实现。请**不要修改当前仓库代码**，只作为实施指导。

## 目标（必须实现）
- 增加 AI 生成 **React Flow 的 nodes/edges JSON** 能力。
- 增加 AI 生成 **draw.io（mxGraph）XML** 能力。
- 用户可导入 **JSON 或 XML** 并生成对应流程图。
- **仅保留 Web 端**，删除所有桌面端/Electron 相关代码。
- 将桌面端有但 Web 端缺失的“好用功能”迁移到 Web 端。
- 增加 **Prompt 输入 / 图片输入（若支持）/ Provider 选择 / API 配置入口**，并可保存配置。

## 范围与约束
- 基于 diagrams.net 开源 Web 版。
- Web-only 构建必须包含桌面端好用功能。
- Provider 配置**持久化**（建议 localStorage）。
- 不在浏览器暴露密钥，建议加入后端代理。

## 总体架构建议
- **Web 前端（draw.io Web）**：编辑器、UI 集成、导入导出、图形渲染。
- **AI 代理服务（推荐）**：统一调用各类 provider，避免在前端暴露密钥。
- **配置存储**：前端 localStorage +（可选）服务端默认配置。

## 必须增加的 UI

### AI 面板 / 弹窗
包含：
- Prompt 文本框
- 图片上传（多图可选，取决于 provider）
- Provider 下拉选择
- 生成按钮（JSON / XML 或格式选择）
- 状态提示 / 错误提示

### 设置面板
包含：
- Provider 列表（增删改）
- 字段：name / type / model / apiKey / baseUrl（视 provider 而定）
- 保存到 localStorage

### 导入与导出
包含：
- JSON / XML 上传
- JSON / XML 粘贴
- 导出 XML（原生）
- 导出 JSON（React Flow 格式）

## Provider 类型
支持的 provider 类型：
- openai
- openai-compatible
- gemini
- anthropic
- openrouter
- groq
- deepseek
- ollama
- custom

最小字段：
- `name`
- `type`
- `model`
- `apiKey`
- `baseUrl`（仅 openai-compatible/custom/部分 provider 需要）

## 数据格式

### React Flow JSON（输入/输出）
推荐格式：
```json
{
  "nodes": [
    {
      "id": "node-1",
      "type": "flow",
      "position": { "x": 100, "y": 80 },
      "data": { "label": "Start", "shape": "terminator" },
      "width": 160,
      "height": 60
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "source": "node-1",
      "target": "node-2",
      "label": "Yes",
      "type": "smoothstep",
      "sourceHandle": "right",
      "targetHandle": "left"
    }
  ]
}
```

### draw.io XML（mxGraph）
- XML 根结构：`<mxfile><diagram>...<mxGraphModel>...`
- 节点：`mxCell` + `geometry`（x/y/width/height）
- 边：`mxCell` + `source/target` + `style`

## JSON ↔ XML 转换

### JSON → XML（必须）
实现转换器，要求：
- 每个节点生成一个 `mxCell`
- 根据 JSON 的 `shape` 映射到 draw.io 的 `style`
- 生成边 `mxCell`，设置 `source/target/label`
- 可选：根据 `sourceHandle/targetHandle` 写入入口/出口样式

### XML → JSON（必须）
实现解析器，要求：
- 读取 `mxGraphModel` cells
- 节点 → React Flow `nodes[]`
- 边 → React Flow `edges[]`
- 保留 label

> 如果前端解析 XML 太复杂，可在后端解析并返回 JSON。

## 导入规则
- 导入 **JSON**：
  1) 校验 JSON
  2) 转换为 XML
  3) 导入 draw.io 画布
- 导入 **XML**：
  1) 解析 XML
  2) 导入 draw.io 画布
  3) 可选：同步生成 JSON 以便导出

## AI 生成流程

### AI → JSON → XML → 画布
1) 用户输入 prompt / 图片
2) AI 返回 React Flow JSON
3) JSON → XML
4) 加载到 draw.io 画布

### AI → XML → 画布（可选）
1) AI 直接返回 XML
2) 校验并加载

> 建议以 JSON 为主输出，稳定性更高。

## Web-only 清理与迁移
在新项目目录中：
- 删除 Electron/desktop 目录及构建脚本
- 删除桌面端菜单/资源/入口
- Web-only CI/构建
- 逐项迁移桌面端“好用功能”到 Web

## 在 draw.io 中的典型改造点
重点位置：
- 编辑器初始化（EditorUi / Graph / Editor）
- 侧边栏/图形库（shape library）
- 菜单与命令（File / Insert / Extras）

需要新增：
- AI 菜单入口与弹窗
- JSON/XML 导入导出入口
- Provider 设置入口

## 后端代理（推荐）
建议新增一个轻量服务：
```
POST /api/flow/json
POST /api/flow/xml
GET  /api/providers
```
职责：
- 统一 Provider 调用
- 密钥注入
- 错误处理与超时控制

## 持久化策略
- 前端保存 Provider 配置（localStorage）
- 可选：服务端 `.env` / `config/providers.json` 作为默认值

## 验证与测试建议
- JSON 导入/导出回环测试
- XML 导入/导出回环测试
- AI 生成 JSON 正确渲染到 draw.io
- 边连接点正确
- 大图性能与响应性

## 主要风险
- draw.io 代码量大，改动必须隔离
- mxGraph XML 容易出错，必须严格校验
- AI 输出格式必须严格约束，避免生成非法图

## 新 agent 交付物清单
- Web-only 的 diagrams.net 版本（删除桌面端）
- AI 面板 + Provider 配置 UI
- JSON/XML 导入导出
- AI 代理服务
- 说明文档
