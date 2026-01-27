const iframe = document.getElementById('diagram-frame');
const promptInput = document.getElementById('prompt-input');
const imageInput = document.getElementById('image-input');
const providerSelect = document.getElementById('provider-select');
const generateBtn = document.getElementById('generate-btn');
const clearPromptBtn = document.getElementById('clear-prompt-btn');
const visionDebugBtn = document.getElementById('vision-debug-btn');
const aiStatus = document.getElementById('ai-status');
const importText = document.getElementById('import-text');
const importFile = document.getElementById('import-file');
const importJsonBtn = document.getElementById('import-json-btn');
const importXmlBtn = document.getElementById('import-xml-btn');
const importStatus = document.getElementById('import-status');
const exportXmlBtn = document.getElementById('export-xml-btn');
const exportJsonBtn = document.getElementById('export-json-btn');
const syncBtn = document.getElementById('sync-btn');
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const closeSettingsBtn = document.getElementById('close-settings-btn');
const visionDebugModal = document.getElementById('vision-debug-modal');
const closeVisionDebugBtn = document.getElementById('close-vision-debug-btn');
const visionDebugImage = document.getElementById('vision-debug-image');
const imageViewerModal = document.getElementById('image-viewer-modal');
const imageViewerViewport = document.getElementById('image-viewer-viewport');
const imageViewerImage = document.getElementById('image-viewer-image');
const imageViewerTitle = document.getElementById('image-viewer-title');
const imageViewerZoomOutBtn = document.getElementById('image-viewer-zoom-out');
const imageViewerZoomInBtn = document.getElementById('image-viewer-zoom-in');
const imageViewerResetBtn = document.getElementById('image-viewer-reset');
const imageViewerDownloadBtn = document.getElementById('image-viewer-download');
const imageViewerCloseBtn = document.getElementById('image-viewer-close');
const visionDebugMeta = document.getElementById('vision-debug-meta');
const languageSelect = document.getElementById('language-select');
const providerStatusRow = document.getElementById('provider-status-row');
const providerStatusText = document.getElementById('provider-status-text');
const providerSettingsShortcut = document.getElementById('provider-settings-shortcut');
const providerFields = document.getElementById('provider-fields');
const providerBaseRow = document.getElementById('provider-base-row');
const providerKey = document.getElementById('provider-key');
const providerBase = document.getElementById('provider-base');
const providerModel = document.getElementById('provider-model');
const providerImageModel = document.getElementById('provider-image-model');
const imageModelEnabledToggle = document.getElementById('image-model-enabled');
const toggleKeyVisibilityBtn = document.getElementById('toggle-key-visibility');
const saveProviderBtn = document.getElementById('save-provider-btn');
const getApiBtn = document.getElementById('get-api-btn');
const providerKeyCache = {};
const imagePreview = document.getElementById('image-preview');
const imagePreviewNote = document.getElementById('image-preview-note');
const criticToggle = document.getElementById('critic-toggle');
const overlayTrimToggle = document.getElementById('overlay-trim-toggle');
const qualityModeSelect = document.getElementById('quality-mode');
const overlayFailuresPanel = document.getElementById('overlay-failures');
const overlayFailuresList = document.getElementById('overlay-failures-list');
const retryOverlaysBtn = document.getElementById('retry-overlays-btn');
const calibrateBtn = document.getElementById('calibrate-btn');
const calibrationModal = document.getElementById('calibration-modal');
const calibCloseBtn = document.getElementById('calib-close-btn');
const calibApplyBtn = document.getElementById('calib-apply-btn');
const calibToggleHistoryBtn = document.getElementById('calib-toggle-history');
const calibViewImageBtn = document.getElementById('calib-view-image');
const calibDownloadImageBtn = document.getElementById('calib-download-image');
const calibHistoryPanel = document.getElementById('calib-history');
const calibHistoryList = document.getElementById('calib-history-list');
const calibViewport = document.getElementById('calib-viewport');
const calibCanvas = document.getElementById('calib-canvas');
const calibImage = document.getElementById('calib-image');
const calibBoxLayer = document.getElementById('calib-box-layer');
const calibPointLayer = document.getElementById('calib-point-layer');
const calibDraft = document.getElementById('calib-draft');
const calibStatus = document.getElementById('calib-status');
const calibSelectionSummary = document.getElementById('calib-selection-summary');
const calibToolSelectBtn = document.getElementById('calib-tool-select');
const calibToolNodeBtn = document.getElementById('calib-tool-node');
const calibToolNewBtn = document.getElementById('calib-tool-new');
const calibToolFgBtn = document.getElementById('calib-tool-fg');
const calibToolBgBtn = document.getElementById('calib-tool-bg');
const calibZoomResetBtn = document.getElementById('calib-zoom-reset');
const calibUndoBtn = document.getElementById('calib-undo');
const calibRedoBtn = document.getElementById('calib-redo');
const calibNodePanel = document.getElementById('calib-node-panel');
const calibNodeId = document.getElementById('calib-node-id');
const calibNodeShape = document.getElementById('calib-node-shape');
const calibNodeText = document.getElementById('calib-node-text');
const calibNodeX = document.getElementById('calib-node-x');
const calibNodeY = document.getElementById('calib-node-y');
const calibNodeW = document.getElementById('calib-node-w');
const calibNodeH = document.getElementById('calib-node-h');
const calibNodeDeleteBtn = document.getElementById('calib-node-delete');
const calibOverlayPanel = document.getElementById('calib-overlay-panel');
const calibOvId = document.getElementById('calib-ov-id');
const calibOvKind = document.getElementById('calib-ov-kind');
const calibOvOwner = document.getElementById('calib-ov-owner');
const calibOvX = document.getElementById('calib-ov-x');
const calibOvY = document.getElementById('calib-ov-y');
const calibOvW = document.getElementById('calib-ov-w');
const calibOvH = document.getElementById('calib-ov-h');
const calibOvPointsMeta = document.getElementById('calib-ov-points-meta');
const calibOvClearPointsBtn = document.getElementById('calib-ov-clear-points');
const calibOvSegmentBtn = document.getElementById('calib-ov-segment');
const calibOvSelectOwnerBtn = document.getElementById('calib-ov-select-owner');
const calibOvDeleteBtn = document.getElementById('calib-ov-delete');
const calibOvPreview = document.getElementById('calib-ov-preview');

const SELECTED_PROVIDER_KEY = 'rd.providers.selected.v1';
const LANGUAGE_KEY = 'rd.language.v1';
const PROVIDER_KEYS_KEY = 'rd.providerKeys.v1';
const CRITIC_ENABLED_KEY = 'rd.critic.enabled.v1';
const OVERLAY_TRIM_ENABLED_KEY = 'rd.overlay.trim.v1';
const QUALITY_MODE_KEY = 'rd.quality.mode.v1';
const IMAGE_MODEL_ENABLED_KEY = 'rd.imageModel.enabled.v1';
const CLIENT_ERROR_ENDPOINT = '/api/client-error';
const CANVAS_XML_KEY = 'rd.canvas.xml.v1';
const CALIB_DB_NAME = 'rd.calibration.tasks.v1';
const CALIB_DB_VERSION = 1;
const CALIB_STORE_TASKS = 'tasks';

const state = {
  iframeReady: false,
  currentXml: '',
  currentJson: null,
  pendingExport: null,
  pendingCanvasLoad: null,
  overlaysCache: {},
  providersServer: {},
  primaryProvider: 'openai-compatible',
  language: 'en',
  scrollGuardActive: false,
  scrollGuardUntil: 0,
  criticEnabled: true,
  overlayTrimEnabled: true,
  qualityMode: 'max',
  imageModelEnabled: false,
  lastStructure: null,
  lastImages: null,
  lastPrompt: '',
  lastProviderType: ''
};

const calibState = {
  open: false,
  currentKey: '',
  currentTask: null,
  tasks: [],
  tool: 'select',
  scale: 1,
  panX: 0,
  panY: 0,
  selection: null,
  drag: null
};

const calibUndo = {
  undo: [],
  redo: [],
  lastSig: '',
  isApplying: false
};

const imageViewerState = {
  open: false,
  src: '',
  filename: '',
  scale: 1,
  panX: 0,
  panY: 0,
  drag: null
};

let autosaveTimer = null;
let lastAutosaveTs = 0;

const sampleJson = {
  nodes: [
    {
      id: 'node-1',
      type: 'flow',
      position: { x: 120, y: 120 },
      data: { label: 'Research Question', shape: 'terminator' },
      width: 180,
      height: 60
    },
    {
      id: 'node-2',
      type: 'flow',
      position: { x: 120, y: 240 },
      data: { label: 'Experiment Design', shape: 'process' },
      width: 200,
      height: 70
    },
    {
      id: 'node-3',
      type: 'flow',
      position: { x: 120, y: 360 },
      data: { label: 'Data Collection', shape: 'data' },
      width: 200,
      height: 70
    },
    {
      id: 'node-4',
      type: 'flow',
      position: { x: 120, y: 480 },
      data: { label: 'Analysis', shape: 'process' },
      width: 180,
      height: 70
    },
    {
      id: 'node-5',
      type: 'flow',
      position: { x: 120, y: 600 },
      data: { label: 'Conclusion', shape: 'terminator' },
      width: 170,
      height: 60
    }
  ],
  edges: [
    {
      id: 'edge-1',
      source: 'node-1',
      target: 'node-2',
      label: 'Hypothesis',
      type: 'smoothstep'
    },
    {
      id: 'edge-2',
      source: 'node-2',
      target: 'node-3',
      label: 'Protocol',
      type: 'smoothstep'
    },
    {
      id: 'edge-3',
      source: 'node-3',
      target: 'node-4',
      label: 'Dataset',
      type: 'smoothstep'
    },
    {
      id: 'edge-4',
      source: 'node-4',
      target: 'node-5',
      label: 'Findings',
      type: 'smoothstep'
    }
  ]
};

const shapeStyleMap = {
  terminator: 'shape=terminator;whiteSpace=wrap;html=1;rounded=1;',
  process: 'shape=process;whiteSpace=wrap;html=1;rounded=0;',
  decision: 'shape=rhombus;whiteSpace=wrap;html=1;',
  data: 'shape=parallelogram;whiteSpace=wrap;html=1;',
  document: 'shape=document;whiteSpace=wrap;html=1;',
  cylinder: 'shape=cylinder;whiteSpace=wrap;html=1;',
  rectangle: 'rounded=0;whiteSpace=wrap;html=1;'
};

const styleShapeMap = {
  terminator: 'terminator',
  process: 'process',
  rhombus: 'decision',
  parallelogram: 'data',
  document: 'document',
  cylinder: 'cylinder'
};

const edgeBaseStyle = 'edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;';

const handlePositions = {
  left: { x: 0, y: 0.5 },
  right: { x: 1, y: 0.5 },
  top: { x: 0.5, y: 0 },
  bottom: { x: 0.5, y: 1 }
};

const i18n = {
  en: {
    brandTitle: 'Research Figure Workbench',
    brandSubtitle: 'AI-generated editable/exportable research figures',
    syncFromCanvas: 'Sync From Canvas',
    exportXml: 'Export XML',
    exportJson: 'Export JSON',
    settings: 'Settings',
    aiGenerate: 'AI Generate',
    promptLabel: 'Prompt',
    promptPlaceholder: 'Describe the research diagram you want...',
    imagesOptional: 'Images (optional)',
    provider: 'Provider',
    outputFormat: 'Output Format',
    generateDiagram: 'Generate Diagram',
    generateRefImage: 'Generate Ref Image',
    clear: 'Clear',
    import: 'Import',
    pasteJsonXml: 'Paste JSON or XML',
    pastePlaceholder: 'Paste JSON or XML here...',
    uploadJsonXml: 'Upload JSON/XML file',
    importJson: 'Import JSON',
    importXml: 'Import XML',
    liveCanvas: 'Live Canvas',
    loadSample: 'Load Sample',
    close: 'Close',
    language: 'Language',
    languageSelect: 'Select language',
    quality: 'Quality',
    qualityMode: 'Quality Mode',
    qualityBalanced: 'Balanced',
    qualityMax: 'Max',
    qualityModeHint: 'Max uses heavier vision/OCR passes for higher fidelity.',
    enableCritic: 'Enable Critic Review',
    criticHint: 'Runs an extra pass comparing the reference image and rendered diagram.',
    overlayTrim: 'Trim Overlay Whitespace',
    overlayTrimHint: 'Tightens extracted overlay bounds to remove blank margins (recommended).',
    providerSettings: 'Provider Settings',
    providerLabelOpenaiCompatible: 'OpenAI-compatible',
    providerLabelOpenai: 'OpenAI',
    providerLabelGemini: 'Gemini',
    providerLabelAnthropic: 'Anthropic',
    providerLabelOpenRouter: 'OpenRouter',
    providerLabelGroq: 'Groq',
    providerLabelDeepSeek: 'DeepSeek',
    providerLabelOllama: 'Ollama (Local)',
    providerLabelCustom: 'Custom',
    getApi: 'Get API',
    enableImageModel: 'Enable Image Model',
    enableImageModelHint: 'When enabled, the app may auto-generate/modify a reference image using the Image Model. Turn off to block all image generation calls.',
    providerLlmModel: 'LLM Model',
    providerLlmModelPlaceholder: 'gpt-4o-mini',
    providerImageModel: 'Image Model',
    providerImageModelPlaceholder: 'gpt-image-1',
    providerModel: 'Model',
    providerModelPlaceholder: 'gpt-4o-mini',
    providerApiKey: 'API Key',
    providerApiKeyPlaceholder: 'Enter API key',
    providerBaseUrl: 'Base URL',
    providerBaseUrlPlaceholder: 'http://localhost:11434/v1',
    saveProvider: 'Save Provider',
    resetForm: 'Reset Form',
    toggleKeyVisibility: 'Show or hide API key',
    providerMissing: 'Provider needs a model name.',
    providerBaseRequired: 'Base URL is required for this provider.',
    providerSaved: 'Provider saved.',
    providerSaveFailed: 'Save failed: {error}',
    providerConfigHint: 'Pick a provider, then fill in required fields.',
    warningImagesNotSupported: 'Images are not supported for this provider/model; ignored images.',
    warningImagesRejected: 'Provider rejected image input; ignored images.',
    providerConfigured: 'Configured',
    providerNotConfigured: 'Not configured',
    providerStatus: '{name} · {status}',
    providerNone: 'No provider',
    activeProvider: 'Active Provider',
    openSettings: 'Open Settings',
    visionDebug: 'Vision Debug',
    visionDebugWorking: 'Analyzing image (debug)...',
    visionDebugFailed: 'Vision debug failed: {error}',
    imageUploadHint: 'Click to upload images (up to 4)',
    imageUploadNote: 'Supported: PNG, JPG, WEBP',
    fileUploadHint: 'Click to upload JSON or XML',
    fileUploadNote: 'Supported: .json, .xml',
    imageNoneSelected: 'No images selected.',
    imageSelectedCount: '{count} image(s) selected.',
    imageMaxKept: 'Only the first {max} images are kept.',
    imagePreprocessing: 'Processing images...',
    imageUnsupportedType: 'Unsupported image type: {type}. Please upload PNG/JPG/WEBP.',
    imageDecodeFailed: 'Failed to decode image: {name}. Please try another image.',
    overlayDetecting: 'Detecting image overlays...',
    overlayExtracting: 'Extracting image overlays...',
    overlayApplied: 'Overlays applied: {count}.',
    overlayFailed: 'Failed overlays: {count}.',
    overlayFailuresTitle: 'Overlay failures',
    retryFailedOverlays: 'Retry failed overlays',
    retryingOverlays: 'Retrying failed overlays...',
    retryOverlaysFailed: 'Retry failed overlays failed: {error}',
    structureDetecting: 'Extracting diagram structure from image...',
    structureRendering: 'Rendering structure to XML...',
    criticReviewing: 'Critic reviewing rendered diagram...',
    criticApplying: 'Applying critic improvements...',
    criticFailed: 'Critic failed: {error}',
    criticApplyConfirm: 'Critic suggested improvements. Apply them to the canvas now? (This will overwrite the current diagram.)',
    idle: 'Idle',
    generatingRefImage: 'Generating reference image...',
    refImageGenerated: 'Reference image generated.',
    refImageFailed: 'Image generation failed: {error}',
    imageModelMissing: 'Image model is not configured. Set it in Settings.',
    imageModelDisabled: 'Image generation is disabled in Settings.',
    selectProviderFirst: 'Select a provider first.',
    enterPromptFirst: 'Enter a prompt first.',
    generatingDiagram: 'Generating diagram...',
    generatingDiagramAttempt: 'Generating diagram (attempt {attempt}/{max})...',
    validatingOutput: 'Validating output...',
    invalidOutput: 'Invalid output: {error}',
    canvasImportError: 'Canvas import error: {error}',
    diagramLoaded: 'Diagram loaded into canvas.',
    generationFailed: 'Generation failed: {error}',
    jsonImported: 'JSON imported into canvas.',
    jsonImportFailed: 'JSON import failed: {error}',
    xmlImported: 'XML imported into canvas.',
    xmlImportFailed: 'XML import failed: {error}',
    syncingFromCanvas: 'Syncing from canvas...',
    canvasSynced: 'Canvas synced to JSON/XML.',
    syncFailed: 'Sync failed: {error}',
    exportingXml: 'Exporting XML...',
    exportingJson: 'Exporting JSON...',
    editorExporting: 'Exporting...',
    editorRendering: 'Rendering...',
    exportFailed: 'Export failed: {error}',
    xmlExported: 'XML exported.',
    jsonExported: 'JSON exported.'
    ,
    precisionCalibrate: 'Precision Calibrate',
    toggleHistory: 'Toggle History',
    viewRefImage: 'View Ref Image',
    downloadRefImage: 'Download Ref Image',
    referenceImage: 'Reference Image',
    zoomIn: 'Zoom In',
    zoomOut: 'Zoom Out',
    download: 'Download',
    imageViewerHint: 'Scroll to zoom, drag to pan. Double-click to reset.',
    applyToCanvas: 'Apply to Canvas',
    history: 'History',
    selection: 'Selection',
    calibSelectHint: 'Select a node or overlay to edit. Tip: Ctrl+click selects the node under the cursor.',
    toolSelect: 'Select',
    toolNewNode: 'New Node',
    toolNewOverlay: 'New Overlay',
    toolFg: 'FG',
    toolBg: 'BG',
    resetView: 'Reset View',
    undo: 'Undo',
    redo: 'Redo',
    node: 'Node',
    nodeId: 'ID',
    nodeShape: 'Shape',
    nodeText: 'Text',
    deleteNode: 'Delete',
    overlay: 'Overlay',
    overlayId: 'ID',
    overlayKind: 'Kind',
    overlayKindIcon: 'Icon',
    overlayKindPhoto: 'Photo',
    overlayKindChart: 'Chart',
    overlayKindPlot: 'Plot',
    overlayKind3d: '3D',
    overlayKindNoise: 'Noise',
    overlayKindScreenshot: 'Screenshot',
    overlayOwner: 'Owner',
    overlayGranularity: 'Granularity',
    bbox: 'bbox',
    granularityAlphaMask: 'alphaMask',
	    granularityOpaqueRect: 'opaqueRect',
	    clearPoints: 'Clear Points',
	    segmentOverlay: 'Segment',
	    selectOwnerNode: 'Select Node',
	    deleteOverlay: 'Delete',
	    legendNode: 'Node (editable)',
	    legendOverlay: 'Overlay (image/icon)',
    historyEmpty: 'No tasks yet.',
    untitledTask: 'Untitled',
    openTask: 'Open',
    deleteTask: 'Delete',
    calibLoaded: 'Task loaded.',
    calibPreparing: 'Opening calibration...',
    calibOpened: 'Calibration opened.',
    calibNeedsImage: 'Upload a reference image first.',
    calibFailed: 'Calibration failed: {error}',
    calibSelectOverlayFirst: 'Select an overlay first.',
    calibOverlayCreated: 'Overlay created. Now click to add FG/BG points.',
    calibNodeCreated: 'Node created.',
    calibSegmenting: 'Segmenting overlay...',
    calibSegmented: 'Overlay segmented.',
    calibSegmentFailed: 'Segment failed: {error}',
    calibApplying: 'Applying to canvas...',
    calibApplied: 'Applied to canvas.',
    calibApplyFailed: 'Apply failed: {error}',
    calibPointsMeta: 'FG: {fg} / BG: {bg}',
    calibIntro: 'Open a task from history, or enter prompt + upload image to create a new one.',
    calibOwnerGlobal: '(global)'
  },
  zh: {
    brandTitle: '科研作图工作台',
    brandSubtitle: 'AI 生成可编辑可导出的科研图',
    syncFromCanvas: '从画布同步',
    exportXml: '导出 XML',
    exportJson: '导出 JSON',
    settings: '设置',
    aiGenerate: 'AI 生成',
    promptLabel: '提示词',
    promptPlaceholder: '描述你要生成的科研图...',
    imagesOptional: '图片（可选）',
    provider: '模型供应商',
    outputFormat: '输出格式',
    generateDiagram: '生成图',
    generateRefImage: 'AI 生图',
    clear: '清空',
    import: '导入',
    pasteJsonXml: '粘贴 JSON 或 XML',
    pastePlaceholder: '在此粘贴 JSON 或 XML...',
    uploadJsonXml: '上传 JSON/XML 文件',
    importJson: '导入 JSON',
    importXml: '导入 XML',
    liveCanvas: '实时画布',
    loadSample: '载入示例',
    close: '关闭',
    language: '语言',
    languageSelect: '选择语言',
    quality: '质量',
    qualityMode: '质量模式',
    qualityBalanced: '均衡',
    qualityMax: '最高质量',
    qualityModeHint: '最高质量会使用更重的视觉/OCR流程以获得更高还原度。',
    enableCritic: '启用终审（Critic）',
    criticHint: '额外执行一次终审，对比参考图与渲染结果以提升准确率。',
    overlayTrim: '覆盖层去白边裁剪',
    overlayTrimHint: '对覆盖层抠图结果进行自动收紧，去除空白边缘（推荐开启）。',
    providerSettings: '模型供应商设置',
    providerLabelOpenaiCompatible: 'OpenAI 兼容',
    providerLabelOpenai: 'OpenAI',
    providerLabelGemini: 'Gemini',
    providerLabelAnthropic: 'Anthropic',
    providerLabelOpenRouter: 'OpenRouter',
    providerLabelGroq: 'Groq',
    providerLabelDeepSeek: 'DeepSeek',
    providerLabelOllama: 'Ollama（本地）',
    providerLabelCustom: '自定义',
    getApi: '获取API',
    enableImageModel: '启用生图模型',
    enableImageModelHint: '开启后会自动判断是否需要调用生图模型生成/修改参考图；关闭后任何情况下都不会调用生图模型。',
    providerLlmModel: 'LLM 模型',
    providerLlmModelPlaceholder: 'gpt-4o-mini',
    providerImageModel: '生图模型',
    providerImageModelPlaceholder: 'gpt-image-1',
    providerModel: '模型',
    providerModelPlaceholder: 'gpt-4o-mini',
    providerApiKey: 'API 密钥',
    providerApiKeyPlaceholder: '输入 API Key',
    providerBaseUrl: '基础地址',
    providerBaseUrlPlaceholder: 'http://localhost:11434/v1',
    saveProvider: '保存',
    resetForm: '重置表单',
    toggleKeyVisibility: '显示或隐藏密钥',
    providerMissing: '请填写模型名称。',
    providerBaseRequired: '该模型供应商需要填写基础地址。',
    providerSaved: '已保存。',
    providerSaveFailed: '保存失败：{error}',
    providerConfigHint: '请选择模型供应商并填写必要字段。',
    warningImagesNotSupported: '该 Provider/模型不支持图片输入，已忽略图片。',
    warningImagesRejected: '模型供应商拒绝了图片输入，已忽略图片。',
    providerConfigured: '已配置',
    providerNotConfigured: '未配置',
    providerStatus: '{name} · {status}',
    providerNone: '暂无供应商',
    activeProvider: '当前供应商',
    openSettings: '打开设置',
    visionDebug: '识别调试',
    visionDebugWorking: '正在分析图片（调试）...',
    visionDebugFailed: '识别调试失败：{error}',
    imageUploadHint: '点击上传图片（最多 4 张）',
    imageUploadNote: '支持 PNG、JPG、WEBP',
    fileUploadHint: '点击上传 JSON 或 XML',
    fileUploadNote: '支持 .json、.xml',
    imageNoneSelected: '未选择图片。',
    imageSelectedCount: '已选择 {count} 张图片。',
    imageMaxKept: '最多保留前 {max} 张图片。',
    imagePreprocessing: '正在处理图片...',
    imageUnsupportedType: '不支持的图片格式：{type}。请上传 PNG/JPG/WEBP。',
    imageDecodeFailed: '图片解码失败：{name}。请更换图片后重试。',
    overlayDetecting: '正在识别图片覆盖层...',
    overlayExtracting: '正在提取图片覆盖层...',
    overlayApplied: '已添加覆盖层：{count} 个。',
    overlayFailed: '覆盖层失败：{count} 个。',
    overlayFailuresTitle: '覆盖层失败列表',
    retryFailedOverlays: '重试失败覆盖层',
    retryingOverlays: '正在重试失败覆盖层...',
    retryOverlaysFailed: '重试失败覆盖层失败：{error}',
    structureDetecting: '正在从图片识别结构...',
    structureRendering: '正在将结构渲染为 XML...',
    criticReviewing: '正在终审渲染结果...',
    criticApplying: '正在应用终审修正...',
    criticFailed: '终审失败：{error}',
    criticApplyConfirm: '终审已给出改进建议，是否立即应用到画布？（将覆盖当前画布内容）',
    idle: '空闲',
    generatingRefImage: '正在生成参考图...',
    refImageGenerated: '参考图已生成。',
    refImageFailed: '生图失败：{error}',
    imageModelMissing: '生图模型未配置，请在设置中填写。',
    imageModelDisabled: '生图功能已在设置中关闭。',
    selectProviderFirst: '请先选择模型供应商。',
    enterPromptFirst: '请先输入提示词。',
    generatingDiagram: '正在生成...',
    generatingDiagramAttempt: '正在生成（第 {attempt}/{max} 次）...',
    validatingOutput: '正在校验输出...',
    invalidOutput: '输出无效：{error}',
    canvasImportError: '画布导入出错：{error}',
    diagramLoaded: '已加载到画布。',
    generationFailed: '生成失败：{error}',
    jsonImported: 'JSON 已导入画布。',
    jsonImportFailed: 'JSON 导入失败：{error}',
    xmlImported: 'XML 已导入画布。',
    xmlImportFailed: 'XML 导入失败：{error}',
    syncingFromCanvas: '正在从画布同步...',
    canvasSynced: '已同步 JSON/XML。',
    syncFailed: '同步失败：{error}',
    exportingXml: '正在导出 XML...',
    exportingJson: '正在导出 JSON...',
    editorExporting: '正在导出...',
    editorRendering: '正在渲染...',
    exportFailed: '导出失败：{error}',
    xmlExported: 'XML 已导出。',
    jsonExported: 'JSON 已导出。'
    ,
    precisionCalibrate: '精确校对',
    toggleHistory: '历史',
    viewRefImage: '查看参考图',
    downloadRefImage: '下载参考图',
    referenceImage: '参考图',
    zoomIn: '放大',
    zoomOut: '缩小',
    download: '下载',
    imageViewerHint: '滚轮缩放，拖拽平移，双击重置。',
    applyToCanvas: '应用到画布',
    history: '历史记录',
    selection: '选择',
    calibSelectHint: '请选择一个节点或覆盖层进行编辑。提示：按住 Ctrl 点击可选中被覆盖的节点。',
    toolSelect: '选择',
    toolNewNode: '新建节点',
    toolNewOverlay: '新建覆盖层',
    toolFg: '前景',
    toolBg: '背景',
    resetView: '重置视图',
    undo: '撤销',
    redo: '重做',
    node: '节点',
    nodeId: 'ID',
    nodeShape: '形状',
    nodeText: '文字',
    deleteNode: '删除',
    overlay: '覆盖层',
    overlayId: 'ID',
    overlayKind: '类型',
    overlayKindIcon: '图标',
    overlayKindPhoto: '照片',
    overlayKindChart: '图表',
    overlayKindPlot: '曲线图',
    overlayKind3d: '3D 示意',
    overlayKindNoise: '噪声/纹理',
    overlayKindScreenshot: '截图',
    overlayOwner: '归属',
    overlayGranularity: '粒度',
    bbox: '边界框',
    granularityAlphaMask: '透明抠图',
	    granularityOpaqueRect: '矩形截图',
	    clearPoints: '清空点',
	    segmentOverlay: '分割',
	    selectOwnerNode: '选中所属节点',
	    deleteOverlay: '删除',
	    legendNode: '节点（可编辑）',
	    legendOverlay: '覆盖层（图片/图标）',
    historyEmpty: '暂无历史任务。',
    untitledTask: '未命名任务',
    openTask: '打开',
    deleteTask: '删除',
    calibLoaded: '任务已载入。',
    calibPreparing: '正在打开校对...',
    calibOpened: '已打开校对。',
    calibNeedsImage: '请先上传参考图。',
    calibFailed: '校对失败：{error}',
    calibSelectOverlayFirst: '请先选择一个覆盖层。',
    calibOverlayCreated: '覆盖层已创建，请继续点选前景/背景点。',
    calibNodeCreated: '节点已创建。',
    calibSegmenting: '正在分割覆盖层...',
    calibSegmented: '覆盖层已分割。',
    calibSegmentFailed: '分割失败：{error}',
    calibApplying: '正在应用到画布...',
    calibApplied: '已应用到画布。',
    calibApplyFailed: '应用失败：{error}',
    calibPointsMeta: '前景点：{fg} · 背景点：{bg}',
    calibIntro: '从历史记录中打开任务，或输入提示词并上传图片来创建新任务。',
    calibOwnerGlobal: '（全局）'
  }
};

const providerCatalog = [
  { type: 'openai-compatible', labelKey: 'providerLabelOpenaiCompatible', label: 'OpenAI-compatible', requiresBase: true, defaultModel: '', defaultBase: 'https://0-0.pro/v1' },
  { type: 'openai', labelKey: 'providerLabelOpenai', label: 'OpenAI', requiresBase: false, defaultModel: 'gpt-4o-mini', defaultBase: 'https://api.openai.com/v1' },
  { type: 'gemini', labelKey: 'providerLabelGemini', label: 'Gemini', requiresBase: false, defaultModel: 'gemini-2.5-flash', defaultBase: '' },
  { type: 'anthropic', labelKey: 'providerLabelAnthropic', label: 'Anthropic', requiresBase: false, defaultModel: 'claude-3-5-sonnet-20241022', defaultBase: '' },
  { type: 'openrouter', labelKey: 'providerLabelOpenRouter', label: 'OpenRouter', requiresBase: true, defaultModel: 'openai/gpt-4o-mini', defaultBase: 'https://openrouter.ai/api/v1' },
  { type: 'groq', labelKey: 'providerLabelGroq', label: 'Groq', requiresBase: true, defaultModel: 'llama-3.1-70b-versatile', defaultBase: 'https://api.groq.com/openai/v1' },
  { type: 'deepseek', labelKey: 'providerLabelDeepSeek', label: 'DeepSeek', requiresBase: true, defaultModel: 'deepseek-chat', defaultBase: 'https://api.deepseek.com/v1' },
  { type: 'ollama', labelKey: 'providerLabelOllama', label: 'Ollama (Local)', requiresBase: true, defaultModel: 'llama3.1', defaultBase: 'http://localhost:11434/v1' },
  { type: 'custom', labelKey: 'providerLabelCustom', label: 'Custom', requiresBase: true, defaultModel: '', defaultBase: '' }
];

const providerCatalogByType = providerCatalog.reduce((acc, item) => {
  acc[item.type] = item;
  return acc;
}, {});

function t(key, vars = {}) {
  const bundle = i18n[state.language] || i18n.en;
  const template = bundle[key] || i18n.en[key] || key;
  return template.replace(/\{(\w+)\}/g, (_, token) => vars[token] ?? '');
}

function loadProviderKeyCache() {
  try {
    const raw = localStorage.getItem(PROVIDER_KEYS_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    Object.keys(parsed).forEach((key) => {
      providerKeyCache[key] = String(parsed[key] ?? '');
    });
  } catch (err) {
    // ignore
  }
}

function persistProviderKeyCache() {
  try {
    localStorage.setItem(PROVIDER_KEYS_KEY, JSON.stringify(providerKeyCache));
  } catch (err) {
    // ignore
  }
}

function fileKey(file) {
  return `${file.name}|${file.size}|${file.lastModified}`;
}

const imageUrlCache = new Map();

function updateImageInputFiles(files) {
  const dt = new DataTransfer();
  files.forEach((file) => dt.items.add(file));
  imageInput.files = dt.files;
}

async function dataUrlToFile(dataUrl, filename) {
  const name = String(filename || '').trim() || 'generated.png';
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  const type = blob.type || 'image/png';
  return new File([blob], name, { type, lastModified: Date.now() });
}

function renderImagePreviews(messageOverride) {
  if (!imagePreview || !imagePreviewNote) return;
  const files = Array.from(imageInput.files || []);
  if (visionDebugBtn) visionDebugBtn.disabled = files.length === 0;
  if (calibrateBtn) calibrateBtn.disabled = false;
  const keys = new Set(files.map(fileKey));

  for (const [key, url] of imageUrlCache.entries()) {
    if (!keys.has(key)) {
      URL.revokeObjectURL(url);
      imageUrlCache.delete(key);
    }
  }

  imagePreview.innerHTML = '';
  if (files.length === 0) {
    imagePreviewNote.textContent = messageOverride || t('imageNoneSelected');
    return;
  }

  imagePreviewNote.textContent = messageOverride || t('imageSelectedCount', { count: String(files.length) });

  files.forEach((file, index) => {
    const key = fileKey(file);
    const url = imageUrlCache.get(key) || URL.createObjectURL(file);
    imageUrlCache.set(key, url);

    const item = document.createElement('div');
    item.className = 'thumb';
    item.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openImageViewer(url, file.name);
    });

    const img = document.createElement('img');
    img.src = url;
    img.alt = file.name;
    img.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openImageViewer(url, file.name);
    });

    const caption = document.createElement('div');
    caption.className = 'thumb-caption';
    caption.textContent = file.name;

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'thumb-remove';
    remove.innerHTML = '&times;';
    remove.title = t('clear');
    remove.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      const next = Array.from(imageInput.files || []).filter((_, idx) => idx !== index);
      updateImageInputFiles(next);
      renderImagePreviews();
    });

    item.append(img, remove, caption);
    imagePreview.appendChild(item);
  });
}

function openCalibration() {
  if (!calibrationModal) return;
  calibrationModal.setAttribute('aria-hidden', 'false');
  calibState.open = true;
  resetCalibView();
  renderCalibBoxes();
  renderCalibPoints();
  updateCalibUndoButtons();
}

function closeCalibration() {
  if (!calibrationModal) return;
  calibrationModal.setAttribute('aria-hidden', 'true');
  calibState.open = false;
}

let calibDb = null;
let calibDbPromise = null;

function resetCalibDb() {
  try {
    if (calibDb) calibDb.close();
  } catch (err) {
    // ignore
  }
  calibDb = null;
  calibDbPromise = null;
}

function idbRequest(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('IndexedDB error'));
  });
}

async function openCalibDb() {
  if (calibDb) return calibDb;
  if (calibDbPromise) return calibDbPromise;
  if (!('indexedDB' in window)) throw new Error('IndexedDB is not available.');
  calibDbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(CALIB_DB_NAME, CALIB_DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(CALIB_STORE_TASKS)) {
        const store = db.createObjectStore(CALIB_STORE_TASKS, { keyPath: 'key' });
        store.createIndex('updatedAt', 'updatedAt', { unique: false });
      }
    };
    req.onblocked = () => {
      // Another tab may be holding an old connection open. We'll retry on demand.
      try {
        console.warn('IndexedDB open blocked.');
      } catch (err) {
        // ignore
      }
    };
    req.onsuccess = () => {
      const db = req.result;
      calibDb = db;
      calibDbPromise = null;
      try {
        db.onversionchange = () => {
          resetCalibDb();
        };
      } catch (err) {
        // ignore
      }
      resolve(db);
    };
    req.onerror = () => {
      calibDbPromise = null;
      reject(req.error || new Error('Failed to open IndexedDB'));
    };
  });
  return calibDbPromise;
}

function isIdbClosingError(err) {
  const name = String(err?.name || '');
  if (name !== 'InvalidStateError') return false;
  const msg = String(err?.message || '').toLowerCase();
  return msg.includes('database connection is closing') || msg.includes('connection is closing');
}

async function withCalibStore(mode, fn) {
  for (let attempt = 1; attempt <= 2; attempt += 1) {
    const db = await openCalibDb();
    try {
      const tx = db.transaction(CALIB_STORE_TASKS, mode);
      const store = tx.objectStore(CALIB_STORE_TASKS);
      // eslint-disable-next-line no-await-in-loop
      return await fn(store, tx);
    } catch (err) {
      if (attempt === 1 && isIdbClosingError(err)) {
        resetCalibDb();
        // eslint-disable-next-line no-continue
        continue;
      }
      throw err;
    }
  }
  throw new Error('IndexedDB unavailable.');
}

async function listCalibTasks() {
  return await withCalibStore('readonly', async (store) => {
    let req;
    try {
      req = store.index('updatedAt').getAll();
    } catch (err) {
      req = store.getAll();
    }
    const all = await idbRequest(req);
    const tasks = Array.isArray(all) ? all : [];
    tasks.sort((a, b) => Number(b.updatedAt || 0) - Number(a.updatedAt || 0));
    return tasks;
  });
}

async function getCalibTask(key) {
  return await withCalibStore('readonly', async (store) => {
    return await idbRequest(store.get(String(key || '')));
  });
}

async function putCalibTask(task) {
  await withCalibStore('readwrite', async (store) => {
    await idbRequest(store.put(task));
  });
}

async function deleteCalibTask(key) {
  await withCalibStore('readwrite', async (store) => {
    await idbRequest(store.delete(String(key || '')));
  });
}

async function sha256Hex(buffer) {
  let data;
  if (buffer instanceof ArrayBuffer) {
    data = buffer;
  } else if (buffer && ArrayBuffer.isView(buffer)) {
    data = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  } else {
    throw new Error('sha256Hex expects ArrayBuffer or ArrayBufferView.');
  }
  const hash = await crypto.subtle.digest('SHA-256', data);
  const bytes = new Uint8Array(hash);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

async function computeTaskKey(image, prompt) {
  const buf = await (await fetch(image.dataUrl)).arrayBuffer();
  const imageHash = await sha256Hex(buf);
  const promptHash = await sha256Hex(new TextEncoder().encode(String(prompt || '')));
  return { key: `${imageHash}|${promptHash}`, imageHash, promptHash };
}

function formatTs(ts) {
  try {
    return new Date(Number(ts || 0)).toLocaleString();
  } catch (err) {
    return '';
  }
}

function truncate(text, max = 64) {
  const s = String(text || '');
  if (s.length <= max) return s;
  return `${s.slice(0, Math.max(0, max - 3))}...`;
}

function renderCalibHistory(tasks) {
  if (!calibHistoryList) return;
  calibHistoryList.innerHTML = '';
  const list = Array.isArray(tasks) ? tasks : [];
  if (list.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'note';
    empty.textContent = t('historyEmpty');
    calibHistoryList.appendChild(empty);
    return;
  }

  list.forEach((task) => {
    const item = document.createElement('div');
    item.className = 'history-item';

    const left = document.createElement('div');

    const title = document.createElement('div');
    title.className = 'history-item-title';
    title.textContent = truncate(task.prompt || '', 52) || t('untitledTask');

    const meta = document.createElement('div');
    meta.className = 'history-item-meta';
    meta.textContent = formatTs(task.updatedAt || task.createdAt || 0);

    left.append(title, meta);

    const actions = document.createElement('div');
    actions.className = 'history-item-actions';

    const openBtn = document.createElement('button');
    openBtn.type = 'button';
    openBtn.className = 'ghost small';
    openBtn.textContent = t('openTask');
    openBtn.addEventListener('click', async () => {
      const full = await getCalibTask(task.key);
      if (full) {
        await openCalibTask(full);
      }
    });

    const delBtn = document.createElement('button');
    delBtn.type = 'button';
    delBtn.className = 'ghost small';
    delBtn.textContent = t('deleteTask');
    delBtn.addEventListener('click', async () => {
      await deleteCalibTask(task.key);
      await refreshCalibHistory();
    });

    actions.append(openBtn, delBtn);
    item.append(left, actions);
    calibHistoryList.appendChild(item);
  });
}

async function refreshCalibHistory() {
  try {
    calibState.tasks = await listCalibTasks();
  } catch (err) {
    calibState.tasks = [];
  }
  renderCalibHistory(calibState.tasks);
}

let calibShapeCatalogLoaded = false;
let calibShapeCatalog = [];

const CALIB_COMMON_SHAPES = ['roundRect', 'rect', 'ellipse', 'rhombus', 'cloud', 'cylinder', 'parallelogram', 'trapezoid'];

const SHAPE_LABELS = {
  roundRect: { en: 'Rounded rectangle', zh: '圆角矩形' },
  rect: { en: 'Rectangle', zh: '矩形' },
  ellipse: { en: 'Ellipse', zh: '椭圆' },
  rhombus: { en: 'Diamond', zh: '菱形' },
  cloud: { en: 'Cloud', zh: '云形' },
  cylinder: { en: 'Cylinder', zh: '圆柱' },
  parallelogram: { en: 'Parallelogram', zh: '平行四边形' },
  trapezoid: { en: 'Trapezoid', zh: '梯形' },
  label: { en: 'Label', zh: '文本' }
};

function shapeDisplayName(shapeId) {
  const key = String(shapeId || '').trim();
  if (!key) return '';
  const entry = SHAPE_LABELS[key];
  if (entry && entry[state.language]) return entry[state.language];
  return key;
}

function shapeOptionText(shapeId) {
  const id = String(shapeId || '').trim();
  if (!id) return '';
  const name = shapeDisplayName(id);
  if (state.language === 'zh') return name || id;
  if (name && name !== id) return `${name} (${id})`;
  return id;
}

function ensureSelectOption(selectEl, value) {
  if (!selectEl) return;
  const val = String(value || '').trim();
  if (!val) return;
  const exists = Array.from(selectEl.options || []).some((o) => String(o.value) === val);
  if (exists) return;
  const opt = document.createElement('option');
  opt.value = val;
  opt.textContent = shapeOptionText(val);
  selectEl.appendChild(opt);
}

function renderCalibShapeOptions() {
  if (!calibNodeShape) return;
  const current = String(calibNodeShape.value || '').trim();
  const shapes = Array.isArray(calibShapeCatalog) && calibShapeCatalog.length ? calibShapeCatalog : CALIB_COMMON_SHAPES;
  const top = CALIB_COMMON_SHAPES.filter((s) => shapes.includes(s));
  const rest = shapes.filter((s) => !top.includes(s));

  calibNodeShape.innerHTML = '';
  [...top, ...rest].slice(0, 20000).forEach((shape) => {
    const opt = document.createElement('option');
    opt.value = String(shape);
    opt.textContent = shapeOptionText(shape);
    calibNodeShape.appendChild(opt);
  });

  if (current) {
    ensureSelectOption(calibNodeShape, current);
    calibNodeShape.value = current;
  }
}

async function ensureCalibShapeCatalogLoaded() {
  if (calibShapeCatalogLoaded) return;
  if (!calibNodeShape) return;
  try {
    const res = await fetch('/api/shapes/catalog', { method: 'GET' });
    if (!res.ok) throw new Error(await readErrorMessage(res));
    const data = await res.json();
    const shapes = Array.isArray(data.shapes) ? data.shapes : [];
    calibShapeCatalog = shapes.slice(0, 20000).map((s) => String(s));
    calibShapeCatalogLoaded = true;
    renderCalibShapeOptions();
  } catch (err) {
    calibShapeCatalog = CALIB_COMMON_SHAPES.slice();
    calibShapeCatalogLoaded = true;
    renderCalibShapeOptions();
  }
}

function setCalibStatus(text, variant = 'idle') {
  if (!calibStatus) return;
  setStatus(calibStatus, text, variant);
}

async function openCalibTask(task) {
  calibState.currentTask = task || null;
  calibState.currentKey = task?.key || '';
  try {
    const imgW = Number(task?.image?.width || 0);
    const imgH = Number(task?.image?.height || 0);
    const metaW = Number(task?.meta?.imageWidth || 0);
    const metaH = Number(task?.meta?.imageHeight || 0);
    if (task && task.structure && imgW > 0 && imgH > 0 && metaW > 0 && metaH > 0 && (imgW !== metaW || imgH !== metaH)) {
      task.structure = rescaleStructureCoordinates(task.structure, { width: metaW, height: metaH }, { width: imgW, height: imgH });
      task.meta = { ...(task.meta || {}), imageWidth: imgW, imageHeight: imgH, rescaledFrom: { imageWidth: metaW, imageHeight: metaH } };
      await putCalibTask(task);
    }
  } catch (err) {
    // ignore
  }
  resetCalibUndo();
  setCalibStatus(t('calibLoaded'), 'success');
  if (calibImage && task?.image?.dataUrl) {
    calibImage.src = task.image.dataUrl;
    if (task?.image?.width && task?.image?.height) {
      calibImage.style.width = `${task.image.width}px`;
      calibImage.style.height = `${task.image.height}px`;
    }
  }
  if (calibCanvas && task?.image?.width && task?.image?.height) {
    calibCanvas.style.width = `${task.image.width}px`;
    calibCanvas.style.height = `${task.image.height}px`;
  }
  updateCalibImageActions();
  setCalibTool('select');
  setSelection(null);
  resetCalibView();
  renderCalibShapeOptions();
  ensureCalibShapeCatalogLoaded();
  await refreshCalibHistory();
  openCalibration();
}

function downloadUrlAsFile(url, filename) {
  const href = String(url || '').trim();
  if (!href) return;
  const name = String(filename || '').trim() || 'image.png';
  const link = document.createElement('a');
  link.href = href;
  link.download = name;
  link.rel = 'noopener';
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function updateCalibImageActions() {
  const img = calibState.currentTask?.image;
  const ok = Boolean(img && typeof img.dataUrl === 'string' && img.dataUrl.startsWith('data:image/'));
  if (calibViewImageBtn) calibViewImageBtn.disabled = !ok;
  if (calibDownloadImageBtn) calibDownloadImageBtn.disabled = !ok;
}

function updateImageViewerTransform() {
  if (!imageViewerImage) return;
  imageViewerImage.style.transform = `translate(${imageViewerState.panX}px, ${imageViewerState.panY}px) scale(${imageViewerState.scale})`;
}

function resetImageViewerView() {
  imageViewerState.scale = 1;
  imageViewerState.panX = 0;
  imageViewerState.panY = 0;
  imageViewerState.drag = null;
  updateImageViewerTransform();
}

function closeImageViewer() {
  if (!imageViewerModal) return;
  imageViewerModal.setAttribute('aria-hidden', 'true');
  imageViewerState.open = false;
  imageViewerState.src = '';
  imageViewerState.filename = '';
  if (imageViewerImage) imageViewerImage.src = '';
  if (imageViewerDownloadBtn) imageViewerDownloadBtn.disabled = true;
}

function openImageViewer(src, filename) {
  if (!imageViewerModal || !imageViewerImage) return;
  const href = String(src || '').trim();
  if (!href) return;
  imageViewerState.open = true;
  imageViewerState.src = href;
  imageViewerState.filename = String(filename || '').trim();
  if (imageViewerTitle) imageViewerTitle.textContent = imageViewerState.filename || t('referenceImage');
  imageViewerModal.setAttribute('aria-hidden', 'false');
  if (imageViewerDownloadBtn) imageViewerDownloadBtn.disabled = false;
  imageViewerImage.src = href;
  resetImageViewerView();
}

function zoomImageViewer(factor, centerX = 0, centerY = 0) {
  const oldScale = clampNumber(imageViewerState.scale, 0.1, 8);
  const nextScale = clampNumber(oldScale * factor, 0.1, 8);
  if (nextScale === oldScale) return;
  const ratio = nextScale / oldScale;
  imageViewerState.panX = ratio * imageViewerState.panX + (1 - ratio) * centerX;
  imageViewerState.panY = ratio * imageViewerState.panY + (1 - ratio) * centerY;
  imageViewerState.scale = nextScale;
  updateImageViewerTransform();
}

function handleImageViewerWheel(event) {
  if (!imageViewerState.open) return;
  if (!imageViewerViewport) return;
  event.preventDefault();
  const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
  const rect = imageViewerViewport.getBoundingClientRect();
  const cx = event.clientX - rect.left - rect.width / 2;
  const cy = event.clientY - rect.top - rect.height / 2;
  zoomImageViewer(factor, cx, cy);
}

function handleImageViewerPointerDown(event) {
  if (!imageViewerState.open) return;
  if (!imageViewerViewport) return;
  if (event.button !== 0) return;
  imageViewerState.drag = {
    startX: event.clientX,
    startY: event.clientY,
    panX: imageViewerState.panX,
    panY: imageViewerState.panY,
    pointerId: event.pointerId
  };
  try {
    imageViewerViewport.setPointerCapture(event.pointerId);
  } catch (err) {
    // ignore
  }
}

function handleImageViewerPointerMove(event) {
  if (!imageViewerState.open) return;
  const drag = imageViewerState.drag;
  if (!drag) return;
  imageViewerState.panX = drag.panX + (event.clientX - drag.startX);
  imageViewerState.panY = drag.panY + (event.clientY - drag.startY);
  updateImageViewerTransform();
}

function handleImageViewerPointerUp(event) {
  if (!imageViewerState.open) return;
  const drag = imageViewerState.drag;
  if (!drag) return;
  if (imageViewerViewport) {
    try {
      imageViewerViewport.releasePointerCapture(drag.pointerId);
    } catch (err) {
      // ignore
    }
  }
  imageViewerState.drag = null;
}

async function openCalibrationFromCurrentInput() {
  openCalibration();
  await refreshCalibHistory();
  renderCalibShapeOptions();
  ensureCalibShapeCatalogLoaded();

  const prompt = promptInput.value.trim();
  const files = Array.from(imageInput.files || []);
  if (!prompt || files.length === 0) {
    setCalibStatus(t('calibIntro'), 'idle');
    if (calibSelectionSummary) calibSelectionSummary.textContent = t('calibSelectHint');
    return;
  }

  try {
    setStatus(aiStatus, t('calibPreparing'), 'working');
    setCalibStatus(t('calibPreparing'), 'working');

    const selected = getSelectedProvider();
    const images = await readImages();
    const first = images[0];
    const { key, imageHash, promptHash } = await computeTaskKey(first, prompt);

	    let task = null;
	    try {
	      task = await getCalibTask(key);
	    } catch (err) {
	      reportClientError(err, { phase: 'calibration.db.get' });
	      task = null;
	    }
	    if (!task) {
	      setStatus(aiStatus, t('structureDetecting'), 'working');
	      setCalibStatus(t('structureDetecting'), 'working');
	      const out = await fetchStructure(images, selected ? selected.type : '', prompt);
	      const imageUsed = out && out.image && typeof out.image === 'object' ? out.image : first;
	      task = {
	        key,
	        imageHash,
	        promptHash,
	        prompt,
	        image: imageUsed,
	        structure: out.structure,
	        meta: out.meta || null,
	        createdAt: Date.now(),
	        updatedAt: Date.now()
	      };
	      try {
	        await putCalibTask(task);
	      } catch (err) {
	        reportClientError(err, { phase: 'calibration.db.put' });
	      }
	    }

    await openCalibTask(task);
    setStatus(aiStatus, t('idle'), 'idle');
  } catch (err) {
    reportClientError(err, { phase: 'calibration.open' });
    setStatus(aiStatus, t('calibFailed', { error: err.message || String(err) }), 'error');
    setCalibStatus(t('calibFailed', { error: err.message || String(err) }), 'error');
  }
}

function setCalibTool(tool) {
  const next = String(tool || 'select');
  calibState.tool = next;
  const map = [
    [calibToolSelectBtn, 'select'],
    [calibToolNodeBtn, 'newNode'],
    [calibToolNewBtn, 'newOverlay'],
    [calibToolFgBtn, 'fg'],
    [calibToolBgBtn, 'bg']
  ];
  map.forEach(([btn, id]) => {
    if (!btn) return;
    btn.classList.toggle('active', id === next);
  });
}

function clampNumber(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, n));
}

function boxFrom(bbox) {
  if (!bbox || typeof bbox !== 'object') return null;
  const x = Number(bbox.x ?? bbox.left ?? 0);
  const y = Number(bbox.y ?? bbox.top ?? 0);
  const w = Number(bbox.w ?? bbox.width ?? 0);
  const h = Number(bbox.h ?? bbox.height ?? 0);
  return { x, y, w, h };
}

function boxIou(a, b) {
  const aa = boxFrom(a);
  const bb = boxFrom(b);
  if (!aa || !bb) return 0;
  const x1 = Math.max(aa.x, bb.x);
  const y1 = Math.max(aa.y, bb.y);
  const x2 = Math.min(aa.x + aa.w, bb.x + bb.w);
  const y2 = Math.min(aa.y + aa.h, bb.y + bb.h);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  const union = aa.w * aa.h + bb.w * bb.h - inter;
  if (union <= 0) return 0;
  return inter / union;
}

function applyCalibTransform() {
  if (!calibCanvas) return;
  calibCanvas.style.transform = `translate(${calibState.panX}px, ${calibState.panY}px) scale(${calibState.scale})`;
}

function resetCalibView() {
  if (!calibViewport || !calibState.currentTask?.image?.width || !calibState.currentTask?.image?.height) return;
  const vw = calibViewport.clientWidth;
  const vh = calibViewport.clientHeight;
  const iw = Number(calibState.currentTask.image.width);
  const ih = Number(calibState.currentTask.image.height);
  const margin = 22;
  const s = Math.min((vw - margin * 2) / iw, (vh - margin * 2) / ih, 1);
  calibState.scale = Number.isFinite(s) && s > 0 ? s : 1;
  calibState.panX = Math.round((vw - iw * calibState.scale) / 2);
  calibState.panY = Math.round((vh - ih * calibState.scale) / 2);
  applyCalibTransform();
}

function clientToImagePoint(clientX, clientY) {
  if (!calibViewport) return { x: 0, y: 0 };
  const rect = calibViewport.getBoundingClientRect();
  const x = (clientX - rect.left - calibState.panX) / calibState.scale;
  const y = (clientY - rect.top - calibState.panY) / calibState.scale;
  return { x, y };
}

function boxToStyle(el, bbox) {
  if (!el || !bbox) return;
  el.style.left = `${bbox.x}px`;
  el.style.top = `${bbox.y}px`;
  el.style.width = `${bbox.w}px`;
  el.style.height = `${bbox.h}px`;
}

function getStructure() {
  const s = calibState.currentTask?.structure;
  return s && typeof s === 'object' ? s : null;
}

function findNode(structure, nodeId) {
  const nodes = Array.isArray(structure?.nodes) ? structure.nodes : [];
  return nodes.find((n) => n && typeof n === 'object' && String(n.id) === String(nodeId)) || null;
}

function findOverlayBySelection(structure, selection) {
  if (!selection || selection.type !== 'overlay') return null;
  if (selection.scope === 'global') {
    const list = Array.isArray(structure?.overlays) ? structure.overlays : [];
    const ov = list.find((o) => o && typeof o === 'object' && String(o.id) === String(selection.id)) || null;
    return ov ? { scope: 'global', overlay: ov, node: null } : null;
  }
  const node = findNode(structure, selection.nodeId);
  if (!node) return null;
  const list = Array.isArray(node.nodeOverlays) ? node.nodeOverlays : [];
  const ov = list.find((o) => o && typeof o === 'object' && String(o.id) === String(selection.id)) || null;
  return ov ? { scope: 'node', overlay: ov, node } : null;
}

let calibSaveTimer = null;

function scheduleCalibSave() {
  if (!calibState.currentTask) return;
  calibState.currentTask.updatedAt = Date.now();
  if (calibSaveTimer) clearTimeout(calibSaveTimer);
  calibSaveTimer = setTimeout(async () => {
    try {
      await putCalibTask(calibState.currentTask);
      if (calibState.open) await refreshCalibHistory();
    } catch (err) {
      // ignore
    }
  }, 400);
}

function cloneForUndo(value) {
  try {
    if (typeof structuredClone === 'function') {
      return structuredClone(value);
    }
  } catch (err) {
    // ignore
  }
  return JSON.parse(JSON.stringify(value));
}

function updateCalibUndoButtons() {
  if (calibUndoBtn) calibUndoBtn.disabled = calibUndo.undo.length === 0;
  if (calibRedoBtn) calibRedoBtn.disabled = calibUndo.redo.length === 0;
}

function resetCalibUndo() {
  calibUndo.undo = [];
  calibUndo.redo = [];
  calibUndo.lastSig = '';
  updateCalibUndoButtons();
}

function pushCalibUndo(reason) {
  if (calibUndo.isApplying) return;
  const task = calibState.currentTask;
  const structure = getStructure();
  if (!task || !structure) return;
  try {
    calibUndo.undo.push({
      reason: String(reason || ''),
      structure: cloneForUndo(structure),
      selection: calibState.selection ? cloneForUndo(calibState.selection) : null,
      tool: String(calibState.tool || 'select')
    });
    if (calibUndo.undo.length > 30) calibUndo.undo = calibUndo.undo.slice(-30);
    calibUndo.redo = [];
    updateCalibUndoButtons();
  } catch (err) {
    // ignore
  }
}

function selectionExists(structure, selection) {
  if (!structure || !selection) return false;
  if (selection.type === 'node') {
    return Boolean(findNode(structure, selection.id));
  }
  if (selection.type === 'overlay') {
    return Boolean(findOverlayBySelection(structure, selection));
  }
  return false;
}

function applyCalibSnapshot(snapshot) {
  const task = calibState.currentTask;
  if (!task || !snapshot || typeof snapshot !== 'object') return;
  const structure = snapshot.structure && typeof snapshot.structure === 'object' ? snapshot.structure : null;
  if (!structure) return;
  calibUndo.isApplying = true;
  try {
    task.structure = structure;
    calibState.tool = String(snapshot.tool || 'select');
    const sel = snapshot.selection || null;
    calibState.selection = selectionExists(structure, sel) ? sel : null;
  } finally {
    calibUndo.isApplying = false;
  }
  scheduleCalibSave();
  renderCalibBoxes();
  renderCalibPoints();
  syncInspectorFromSelection();
  updateCalibUndoButtons();
}

function performCalibUndo() {
  if (!calibState.open) return;
  if (calibUndo.undo.length === 0) return;
  const task = calibState.currentTask;
  const structure = getStructure();
  if (!task || !structure) return;
  const current = {
    reason: 'redo',
    structure: cloneForUndo(structure),
    selection: calibState.selection ? cloneForUndo(calibState.selection) : null,
    tool: String(calibState.tool || 'select')
  };
  const snap = calibUndo.undo.pop();
  calibUndo.redo.push(current);
  if (calibUndo.redo.length > 30) calibUndo.redo = calibUndo.redo.slice(-30);
  applyCalibSnapshot(snap);
}

function performCalibRedo() {
  if (!calibState.open) return;
  if (calibUndo.redo.length === 0) return;
  const task = calibState.currentTask;
  const structure = getStructure();
  if (!task || !structure) return;
  const current = {
    reason: 'undo',
    structure: cloneForUndo(structure),
    selection: calibState.selection ? cloneForUndo(calibState.selection) : null,
    tool: String(calibState.tool || 'select')
  };
  const snap = calibUndo.redo.pop();
  calibUndo.undo.push(current);
  if (calibUndo.undo.length > 30) calibUndo.undo = calibUndo.undo.slice(-30);
  applyCalibSnapshot(snap);
}

function setSelection(selection) {
  calibState.selection = selection;
  renderCalibBoxes();
  renderCalibPoints();
  syncInspectorFromSelection();
}

function renderCalibBoxes() {
  if (!calibBoxLayer) return;
  const structure = getStructure();
  calibBoxLayer.innerHTML = '';
  if (!structure) return;

  const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
  const overlaysGlobal = Array.isArray(structure.overlays) ? structure.overlays : [];

  nodes.forEach((n) => {
    if (!n || typeof n !== 'object' || !n.bbox) return;
    const el = document.createElement('div');
    el.className = 'calib-box node';
    el.dataset.kind = 'node';
    el.dataset.nodeId = String(n.id || '');
    boxToStyle(el, boxFrom(n.bbox));
    if (calibState.selection?.type === 'node' && String(calibState.selection.id) === String(n.id)) {
      el.classList.add('selected');
      ['nw', 'ne', 'sw', 'se'].forEach((h) => {
        const handle = document.createElement('div');
        handle.className = 'calib-handle';
        handle.dataset.handle = h;
        el.appendChild(handle);
      });
    }
    calibBoxLayer.appendChild(el);

    const nodeOverlays = Array.isArray(n.nodeOverlays) ? n.nodeOverlays : [];
    nodeOverlays.forEach((ov) => {
      if (!ov || typeof ov !== 'object' || !ov.bbox) return;
      const oel = document.createElement('div');
      oel.className = 'calib-box overlay';
      oel.dataset.kind = 'overlay';
      oel.dataset.scope = 'node';
      oel.dataset.nodeId = String(n.id || '');
      oel.dataset.ovId = String(ov.id || '');
      boxToStyle(oel, boxFrom(ov.bbox));
      const isSelected =
        calibState.selection?.type === 'overlay' &&
        calibState.selection.scope === 'node' &&
        String(calibState.selection.nodeId) === String(n.id) &&
        String(calibState.selection.id) === String(ov.id);
      if (isSelected) {
        oel.classList.add('selected');
        ['nw', 'ne', 'sw', 'se'].forEach((h) => {
          const handle = document.createElement('div');
          handle.className = 'calib-handle';
          handle.dataset.handle = h;
          oel.appendChild(handle);
        });
      }
      calibBoxLayer.appendChild(oel);
    });
  });

  overlaysGlobal.forEach((ov) => {
    if (!ov || typeof ov !== 'object' || !ov.bbox) return;
    const el = document.createElement('div');
    el.className = 'calib-box overlay';
    el.dataset.kind = 'overlay';
    el.dataset.scope = 'global';
    el.dataset.ovId = String(ov.id || '');
    boxToStyle(el, boxFrom(ov.bbox));
    const isSelected =
      calibState.selection?.type === 'overlay' &&
      calibState.selection.scope === 'global' &&
      String(calibState.selection.id) === String(ov.id);
    if (isSelected) {
      el.classList.add('selected');
      ['nw', 'ne', 'sw', 'se'].forEach((h) => {
        const handle = document.createElement('div');
        handle.className = 'calib-handle';
        handle.dataset.handle = h;
        el.appendChild(handle);
      });
    }
    calibBoxLayer.appendChild(el);
  });
}

function renderCalibPoints() {
  if (!calibPointLayer) return;
  calibPointLayer.innerHTML = '';
  const structure = getStructure();
  if (!structure || calibState.selection?.type !== 'overlay') return;
  const found = findOverlayBySelection(structure, calibState.selection);
  if (!found) return;
  const ov = found.overlay;
  const fg = Array.isArray(ov.fgPoints) ? ov.fgPoints : [];
  const bg = Array.isArray(ov.bgPoints) ? ov.bgPoints : [];
  fg.forEach((p) => {
    const el = document.createElement('div');
    el.className = 'calib-point fg';
    el.style.left = `${Number(p.x)}px`;
    el.style.top = `${Number(p.y)}px`;
    calibPointLayer.appendChild(el);
  });
  bg.forEach((p) => {
    const el = document.createElement('div');
    el.className = 'calib-point bg';
    el.style.left = `${Number(p.x)}px`;
    el.style.top = `${Number(p.y)}px`;
    calibPointLayer.appendChild(el);
  });
}

function syncInspectorFromSelection() {
  const structure = getStructure();
  if (!structure) return;
  const sel = calibState.selection;
  if (!sel) {
    if (calibNodePanel) calibNodePanel.dataset.hidden = 'true';
    if (calibOverlayPanel) calibOverlayPanel.dataset.hidden = 'true';
    if (calibSelectionSummary) calibSelectionSummary.textContent = t('calibSelectHint');
    return;
  }

  if (sel.type === 'node') {
    const node = findNode(structure, sel.id);
    if (!node) return;
    if (calibOverlayPanel) calibOverlayPanel.dataset.hidden = 'true';
    if (calibNodePanel) calibNodePanel.dataset.hidden = 'false';
    if (calibSelectionSummary) calibSelectionSummary.textContent = `${t('node')}: ${String(node.id || '')}`;
    if (calibNodeId) calibNodeId.value = String(node.id || '');
    if (calibNodeShape) {
      const shapeId = String(node.shapeId || 'roundRect');
      ensureSelectOption(calibNodeShape, shapeId);
      calibNodeShape.value = shapeId;
    }
    if (calibNodeText) calibNodeText.value = String(node.text || '');
    const bb = boxFrom(node.bbox) || { x: 0, y: 0, w: 0, h: 0 };
    if (calibNodeX) calibNodeX.value = String(Math.round(bb.x));
    if (calibNodeY) calibNodeY.value = String(Math.round(bb.y));
    if (calibNodeW) calibNodeW.value = String(Math.round(bb.w));
    if (calibNodeH) calibNodeH.value = String(Math.round(bb.h));
    return;
  }

  const found = findOverlayBySelection(structure, sel);
  if (!found) return;
  if (calibNodePanel) calibNodePanel.dataset.hidden = 'true';
  if (calibOverlayPanel) calibOverlayPanel.dataset.hidden = 'false';
  const ov = found.overlay;
  if (calibSelectionSummary) calibSelectionSummary.textContent = `${t('overlay')}: ${String(ov.id || '')}`;
  if (calibOvId) calibOvId.value = String(ov.id || '');
  if (calibOvKind) calibOvKind.value = String(ov.kind || 'icon');
  const bb = boxFrom(ov.bbox) || { x: 0, y: 0, w: 0, h: 0 };
  if (calibOvX) calibOvX.value = String(Math.round(bb.x));
  if (calibOvY) calibOvY.value = String(Math.round(bb.y));
  if (calibOvW) calibOvW.value = String(Math.round(bb.w));
  if (calibOvH) calibOvH.value = String(Math.round(bb.h));
  const g = String(ov.granularity || 'alphaMask');
  document.querySelectorAll('input[name=\"calib-granularity\"]').forEach((r) => {
    r.checked = String(r.value) === g;
  });
  const fg = Array.isArray(ov.fgPoints) ? ov.fgPoints.length : 0;
  const bg = Array.isArray(ov.bgPoints) ? ov.bgPoints.length : 0;
  if (calibOvPointsMeta) calibOvPointsMeta.textContent = t('calibPointsMeta', { fg: String(fg), bg: String(bg) });
  if (calibOvPreview) calibOvPreview.src = typeof ov.dataUrl === 'string' ? ov.dataUrl : '';

	  if (calibOvOwner) {
	    calibOvOwner.innerHTML = '';
	    const optGlobal = document.createElement('option');
	    optGlobal.value = '__global__';
	    optGlobal.textContent = t('calibOwnerGlobal');
	    calibOvOwner.appendChild(optGlobal);

    const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
    nodes.forEach((n) => {
      if (String(n?.render || '').toLowerCase() === 'text') return;
      const opt = document.createElement('option');
      opt.value = String(n.id || '');
      opt.textContent = String(n.text || n.id || '');
      calibOvOwner.appendChild(opt);
    });

	    calibOvOwner.value = found.scope === 'node' ? String(found.node?.id || '') : '__global__';
	  }
	  if (calibOvSelectOwnerBtn) {
	    calibOvSelectOwnerBtn.disabled = found.scope !== 'node';
	  }
	}

function clampBoxToBounds(bbox, bounds) {
  const bb = boxFrom(bbox) || { x: 0, y: 0, w: 0, h: 0 };
  const b = boxFrom(bounds) || { x: 0, y: 0, w: 0, h: 0 };
  const minSize = 6;
  const x1 = clampNumber(bb.x, b.x, b.x + b.w - minSize);
  const y1 = clampNumber(bb.y, b.y, b.y + b.h - minSize);
  const wMax = Math.max(minSize, b.x + b.w - x1);
  const hMax = Math.max(minSize, b.y + b.h - y1);
  const w1 = clampNumber(bb.w, minSize, wMax);
  const h1 = clampNumber(bb.h, minSize, hMax);
  return { x: Math.round(x1), y: Math.round(y1), w: Math.round(w1), h: Math.round(h1) };
}

function clampBoxToImage(bbox) {
  const iw = Number(calibState.currentTask?.image?.width || 0);
  const ih = Number(calibState.currentTask?.image?.height || 0);
  return clampBoxToBounds(bbox, { x: 0, y: 0, w: iw, h: ih });
}

function applyBboxToSelection(selection, bbox) {
  const structure = getStructure();
  if (!structure || !selection) return;
  if (selection.type === 'node') {
    const node = findNode(structure, selection.id);
    if (!node) return;
    node.bbox = clampBoxToImage(bbox);
    return;
  }
  const found = findOverlayBySelection(structure, selection);
  if (!found) return;
  if (found.scope === 'node' && found.node?.bbox) {
    found.overlay.bbox = clampBoxToBounds(bbox, found.node.bbox);
  } else {
    found.overlay.bbox = clampBoxToImage(bbox);
  }
}

function handleCalibWheel(event) {
  if (!calibState.open || !calibViewport || !calibState.currentTask?.image) return;
  if (!event) return;
  event.preventDefault();
  const delta = Number(event.deltaY || 0);
  const factor = delta > 0 ? 0.92 : 1.08;
  const nextScale = clampNumber(calibState.scale * factor, 0.15, 8);
  const rect = calibViewport.getBoundingClientRect();
  const cx = event.clientX - rect.left;
  const cy = event.clientY - rect.top;
  const imgPt = clientToImagePoint(event.clientX, event.clientY);
  calibState.scale = nextScale;
  calibState.panX = Math.round(cx - imgPt.x * nextScale);
  calibState.panY = Math.round(cy - imgPt.y * nextScale);
  applyCalibTransform();
}

const calibKeys = { space: false };

function isEditableTarget(el) {
  if (!el || !(el instanceof HTMLElement)) return false;
  if (el.isContentEditable) return true;
  const tag = el.tagName;
  if (tag === 'TEXTAREA') return !el.disabled;
  if (tag === 'INPUT') return !el.disabled && !el.readOnly;
  if (tag === 'SELECT') return !el.disabled;
  return false;
}

function handleCalibKey(event, isDown) {
  if (!calibState.open) return;
  if (!event) return;

  if (isDown && !isEditableTarget(document.activeElement)) {
    const key = String(event.key || '');
    if (key === 'Delete' || key === 'Backspace') {
      event.preventDefault();
      if (calibState.selection?.type === 'node') {
        deleteSelectedNode();
      } else if (calibState.selection?.type === 'overlay') {
        deleteSelectedOverlay();
      }
      return;
    }
  }

  if (isDown && (event.ctrlKey || event.metaKey)) {
    const key = String(event.key || '').toLowerCase();
    if (!isEditableTarget(document.activeElement)) {
      if (key === 'z' && !event.shiftKey) {
        event.preventDefault();
        performCalibUndo();
        return;
      }
      if (key === 'y' || (key === 'z' && event.shiftKey)) {
        event.preventDefault();
        performCalibRedo();
        return;
      }
    }
  }

  if (event.code === 'Space') {
    calibKeys.space = Boolean(isDown);
    if (calibViewport) {
      calibViewport.style.cursor = calibKeys.space ? 'grab' : '';
    }
  }
}

function selectionFromElement(el) {
  if (!el) return null;
  const kind = el.dataset.kind;
  if (kind === 'node') {
    return { type: 'node', id: el.dataset.nodeId || '' };
  }
  if (kind === 'overlay') {
    const scope = el.dataset.scope === 'node' ? 'node' : 'global';
    const nodeId = el.dataset.nodeId || '';
    const id = el.dataset.ovId || '';
    return { type: 'overlay', scope, nodeId, id };
  }
  return null;
}

function addPointToSelectedOverlay(point, mode) {
  const structure = getStructure();
  if (!structure) return false;
  if (!calibState.selection || calibState.selection.type !== 'overlay') return false;
  const found = findOverlayBySelection(structure, calibState.selection);
  if (!found) return false;
  const ov = found.overlay;
  pushCalibUndo('addPoint');
  const pt = { x: Math.round(point.x), y: Math.round(point.y) };
  const listKey = mode === 'bg' ? 'bgPoints' : 'fgPoints';
  if (!Array.isArray(ov[listKey])) ov[listKey] = [];
  ov[listKey].push(pt);
  scheduleCalibSave();
  renderCalibPoints();
  syncInspectorFromSelection();
  return true;
}

function createOverlayFromBox(bbox) {
  const structure = getStructure();
  if (!structure) return null;
  pushCalibUndo('createOverlay');
  const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
  let bestNode = null;
  let bestScore = 0;
  nodes.forEach((n) => {
    if (!n || typeof n !== 'object' || !n.bbox) return;
    if (String(n.render || '').toLowerCase() === 'text') return;
    const score = boxIou(n.bbox, bbox);
    if (score > bestScore) {
      bestScore = score;
      bestNode = n;
    }
  });

  const id = `man_${Date.now().toString(36)}_${Math.random().toString(16).slice(2, 6)}`;
  const overlay = {
    id,
    kind: 'icon',
    granularity: 'alphaMask',
    bbox: clampBoxToImage(bbox),
    fgPoints: [],
    bgPoints: [],
    confidence: 1
  };

  let selection = null;
  if (bestNode && bestScore > 0) {
    overlay.bbox = clampBoxToBounds(overlay.bbox, bestNode.bbox);
    if (!Array.isArray(bestNode.nodeOverlays)) bestNode.nodeOverlays = [];
    bestNode.nodeOverlays.push(overlay);
    selection = { type: 'overlay', scope: 'node', nodeId: String(bestNode.id || ''), id };
  } else {
    if (!Array.isArray(structure.overlays)) structure.overlays = [];
    structure.overlays.push(overlay);
    selection = { type: 'overlay', scope: 'global', nodeId: '', id };
  }

  scheduleCalibSave();
  return selection;
}

function createNodeFromBox(bbox) {
  const structure = getStructure();
  if (!structure) return null;
  pushCalibUndo('createNode');
  if (!Array.isArray(structure.nodes)) structure.nodes = [];
  const idSet = new Set(structure.nodes.map((n) => String(n?.id || '')));
  const base = `man_node_${Date.now().toString(36)}_${Math.random().toString(16).slice(2, 6)}`;
  let id = base;
  let i = 2;
  while (idSet.has(id)) {
    id = `${base}_${i}`;
    i += 1;
  }

  const node = {
    id,
    bbox: clampBoxToImage(bbox),
    text: '',
    shapeId: 'roundRect',
    render: 'shape',
    confidence: { bbox: 1, text: 0, shape: 0.7 }
  };
  structure.nodes.push(node);
  scheduleCalibSave();
  return { type: 'node', id };
}

function deleteSelectedNode() {
  const structure = getStructure();
  const sel = calibState.selection;
  if (!structure || !sel || sel.type !== 'node') return;
  const id = String(sel.id || '');
  if (!id) return;

  pushCalibUndo('deleteNode');
  structure.nodes = (Array.isArray(structure.nodes) ? structure.nodes : []).filter((n) => String(n?.id || '') !== id);
  structure.edges = (Array.isArray(structure.edges) ? structure.edges : []).filter(
    (e) => String(e?.source || '') !== id && String(e?.target || '') !== id
  );

  scheduleCalibSave();
  setSelection(null);
  renderCalibBoxes();
}

function handleCalibPointerDown(event) {
  if (!calibState.open || !calibViewport || !calibState.currentTask?.image) return;
  if (!event) return;
  if (event.button === 2) return;

  const isPan = event.button === 1 || (event.button === 0 && calibKeys.space);
  if (isPan) {
    calibState.drag = { kind: 'pan', startX: event.clientX, startY: event.clientY, panX: calibState.panX, panY: calibState.panY };
    try {
      calibViewport.setPointerCapture(event.pointerId);
    } catch (err) {
      // ignore
    }
    return;
  }

  const pt = clientToImagePoint(event.clientX, event.clientY);
  const structure = getStructure();
  if (!structure) return;

  // Ctrl+click: quickly select the smallest node under cursor (useful when overlays cover most of it).
  if (calibState.tool === 'select' && (event.ctrlKey || event.metaKey)) {
    const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
    let best = null;
    let bestArea = Infinity;
    for (const n of nodes) {
      if (!n || typeof n !== 'object' || !n.bbox) continue;
      if (String(n.render || '').toLowerCase() === 'text') continue;
      const bb = boxFrom(n.bbox);
      if (!bb) continue;
      if (pt.x < bb.x || pt.y < bb.y || pt.x > bb.x + bb.w || pt.y > bb.y + bb.h) continue;
      const area = bb.w * bb.h;
      if (area < bestArea) {
        best = n;
        bestArea = area;
      }
    }
    if (best) {
      setSelection({ type: 'node', id: String(best.id || '') });
      return;
    }
  }

  if (calibState.tool === 'fg' || calibState.tool === 'bg') {
    const ok = addPointToSelectedOverlay(pt, calibState.tool);
    if (!ok) setCalibStatus(t('calibSelectOverlayFirst'), 'error');
    return;
  }

  const target = event.target instanceof Element ? event.target : null;
  const boxEl = target ? target.closest('.calib-box') : null;

  if (calibState.tool === 'newOverlay' || calibState.tool === 'newNode') {
    if (event.button !== 0) return;
    calibState.drag = { kind: 'draft', draftType: calibState.tool, start: pt, current: pt };
    if (calibDraft) {
      calibDraft.dataset.hidden = 'false';
      boxToStyle(calibDraft, { x: pt.x, y: pt.y, w: 1, h: 1 });
    }
    try {
      calibViewport.setPointerCapture(event.pointerId);
    } catch (err) {
      // ignore
    }
    return;
  }

  if (!boxEl) {
    setSelection(null);
    return;
  }

  const sel = selectionFromElement(boxEl);
  if (!sel) return;
  if (sel.type === 'overlay' && event.shiftKey && sel.scope === 'node' && sel.nodeId) {
    // Allow selecting the underlying node even when overlays cover most of the area.
    setSelection({ type: 'node', id: String(sel.nodeId) });
    return;
  }
  if (!calibState.selection || JSON.stringify(calibState.selection) !== JSON.stringify(sel)) {
    setSelection(sel);
  }

  const handleEl = target && target.classList.contains('calib-handle') ? target : null;
  const handle = handleEl ? handleEl.dataset.handle : '';
  const objBBox = (() => {
    if (sel.type === 'node') return boxFrom(findNode(structure, sel.id)?.bbox) || null;
    const found = findOverlayBySelection(structure, sel);
    return boxFrom(found?.overlay?.bbox) || null;
  })();
  if (!objBBox) return;

  calibState.drag = {
    kind: handle ? 'resize' : 'move',
    selection: sel,
    handle: handle || '',
    start: pt,
    orig: objBBox,
    el: boxEl,
    undoPushed: false
  };

  try {
    calibViewport.setPointerCapture(event.pointerId);
  } catch (err) {
    // ignore
  }
}

function handleCalibPointerMove(event) {
  if (!calibState.open || !calibState.drag) return;
  const drag = calibState.drag;
  if (!event) return;

  if (drag.kind === 'pan') {
    calibState.panX = Math.round(drag.panX + (event.clientX - drag.startX));
    calibState.panY = Math.round(drag.panY + (event.clientY - drag.startY));
    applyCalibTransform();
    return;
  }

  const pt = clientToImagePoint(event.clientX, event.clientY);

  if (drag.kind === 'draft') {
    drag.current = pt;
    const x1 = Math.min(drag.start.x, pt.x);
    const y1 = Math.min(drag.start.y, pt.y);
    const x2 = Math.max(drag.start.x, pt.x);
    const y2 = Math.max(drag.start.y, pt.y);
    if (calibDraft) boxToStyle(calibDraft, { x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
    return;
  }

  if (drag.kind === 'move' || drag.kind === 'resize') {
    if (!drag.undoPushed) {
      pushCalibUndo('moveResize');
      drag.undoPushed = true;
    }
    const dx = pt.x - drag.start.x;
    const dy = pt.y - drag.start.y;
    let next = { ...drag.orig };

    if (drag.kind === 'move') {
      next.x = drag.orig.x + dx;
      next.y = drag.orig.y + dy;
    } else {
      const h = drag.handle;
      if (h.includes('w')) {
        next.x = drag.orig.x + dx;
        next.w = drag.orig.w - dx;
      }
      if (h.includes('e')) {
        next.w = drag.orig.w + dx;
      }
      if (h.includes('n')) {
        next.y = drag.orig.y + dy;
        next.h = drag.orig.h - dy;
      }
      if (h.includes('s')) {
        next.h = drag.orig.h + dy;
      }
    }

    applyBboxToSelection(drag.selection, next);
    const structure = getStructure();
    const effective =
      drag.selection.type === 'node'
        ? boxFrom(findNode(structure, drag.selection.id)?.bbox)
        : boxFrom(findOverlayBySelection(structure, drag.selection)?.overlay?.bbox);
    if (drag.el && effective) {
      boxToStyle(drag.el, effective);
    }
    renderCalibPoints();
    syncInspectorFromSelection();
  }
}

function handleCalibPointerUp(event) {
  if (!calibState.open || !calibState.drag) return;
  const drag = calibState.drag;
  calibState.drag = null;

  if (drag.kind === 'draft') {
    if (calibDraft) calibDraft.dataset.hidden = 'true';
    const start = drag.start;
    const end = drag.current || drag.start;
    const x1 = Math.min(start.x, end.x);
    const y1 = Math.min(start.y, end.y);
    const x2 = Math.max(start.x, end.x);
    const y2 = Math.max(start.y, end.y);
    const bbox = clampBoxToImage({ x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
    if (bbox.w >= 8 && bbox.h >= 8) {
      if (drag.draftType === 'newNode') {
        const sel = createNodeFromBox(bbox);
        if (sel) {
          setSelection(sel);
          setCalibTool('select');
          setCalibStatus(t('calibNodeCreated'), 'success');
        }
      } else {
        const sel = createOverlayFromBox(bbox);
        if (sel) {
          setSelection(sel);
          setCalibTool('fg');
          setCalibStatus(t('calibOverlayCreated'), 'success');
        }
      }
    }
    return;
  }

  if (drag.kind === 'move' || drag.kind === 'resize') {
    scheduleCalibSave();
    renderCalibBoxes();
    syncInspectorFromSelection();
  }
}

function updateSelectedNodeFromInspector() {
  const structure = getStructure();
  const sel = calibState.selection;
  if (!structure || !sel || sel.type !== 'node') return;
  const node = findNode(structure, sel.id);
  if (!node) return;

  pushCalibUndo('editNode');
  if (calibNodeShape) {
    const raw = String(calibNodeShape.value || '').trim();
    if (raw) node.shapeId = raw;
  }
  if (calibNodeText) node.text = String(calibNodeText.value || '');

  const bb = {
    x: Number(calibNodeX?.value),
    y: Number(calibNodeY?.value),
    w: Number(calibNodeW?.value),
    h: Number(calibNodeH?.value)
  };
  node.bbox = clampBoxToImage(bb);
  scheduleCalibSave();
  renderCalibBoxes();
}

function moveOverlayToOwner(found, ownerValue) {
  const structure = getStructure();
  if (!structure || !found) return null;
  const overlay = found.overlay;
  const currentScope = found.scope;

  if (String(ownerValue || '') === '__global__') {
    if (currentScope === 'global') return { type: 'overlay', scope: 'global', nodeId: '', id: String(overlay.id || '') };
    const node = found.node;
    if (node && Array.isArray(node.nodeOverlays)) {
      node.nodeOverlays = node.nodeOverlays.filter((o) => String(o?.id || '') !== String(overlay.id || ''));
    }
    if (!Array.isArray(structure.overlays)) structure.overlays = [];
    overlay.bbox = clampBoxToImage(overlay.bbox);
    structure.overlays.push(overlay);
    return { type: 'overlay', scope: 'global', nodeId: '', id: String(overlay.id || '') };
  }

  const nextNode = findNode(structure, ownerValue);
  if (!nextNode) return null;
  if (String(nextNode.render || '').toLowerCase() === 'text') return null;

  if (currentScope === 'global') {
    structure.overlays = (Array.isArray(structure.overlays) ? structure.overlays : []).filter(
      (o) => String(o?.id || '') !== String(overlay.id || '')
    );
  } else if (found.node && Array.isArray(found.node.nodeOverlays)) {
    found.node.nodeOverlays = found.node.nodeOverlays.filter((o) => String(o?.id || '') !== String(overlay.id || ''));
  }

  if (!Array.isArray(nextNode.nodeOverlays)) nextNode.nodeOverlays = [];
  overlay.bbox = nextNode.bbox ? clampBoxToBounds(overlay.bbox, nextNode.bbox) : clampBoxToImage(overlay.bbox);
  nextNode.nodeOverlays.push(overlay);
  return { type: 'overlay', scope: 'node', nodeId: String(nextNode.id || ''), id: String(overlay.id || '') };
}

function updateSelectedOverlayFromInspector() {
  const structure = getStructure();
  const sel = calibState.selection;
  if (!structure || !sel || sel.type !== 'overlay') return;
  const found = findOverlayBySelection(structure, sel);
  if (!found) return;
  const ov = found.overlay;

  pushCalibUndo('editOverlay');
  if (calibOvKind) ov.kind = String(calibOvKind.value || ov.kind || 'icon');

  const gran = document.querySelector('input[name=\"calib-granularity\"]:checked')?.value;
  if (gran) ov.granularity = String(gran);

  const bb = {
    x: Number(calibOvX?.value),
    y: Number(calibOvY?.value),
    w: Number(calibOvW?.value),
    h: Number(calibOvH?.value)
  };
  if (found.scope === 'node' && found.node?.bbox) {
    ov.bbox = clampBoxToBounds(bb, found.node.bbox);
  } else {
    ov.bbox = clampBoxToImage(bb);
  }

  if (calibOvOwner) {
    const owner = calibOvOwner.value;
    const movedSel = moveOverlayToOwner(found, owner);
    if (movedSel) {
      setSelection(movedSel);
    }
  }

  scheduleCalibSave();
  renderCalibBoxes();
  renderCalibPoints();
  syncInspectorFromSelection();
}

function clearSelectedOverlayPoints() {
  const structure = getStructure();
  const sel = calibState.selection;
  if (!structure || !sel || sel.type !== 'overlay') return;
  const found = findOverlayBySelection(structure, sel);
  if (!found) return;
  pushCalibUndo('clearPoints');
  found.overlay.fgPoints = [];
  found.overlay.bgPoints = [];
  scheduleCalibSave();
  renderCalibPoints();
  syncInspectorFromSelection();
}

function deleteSelectedOverlay() {
  const structure = getStructure();
  const sel = calibState.selection;
  if (!structure || !sel || sel.type !== 'overlay') return;
  const found = findOverlayBySelection(structure, sel);
  if (!found) return;
  pushCalibUndo('deleteOverlay');
  if (found.scope === 'global') {
    structure.overlays = (Array.isArray(structure.overlays) ? structure.overlays : []).filter((o) => String(o?.id || '') !== String(found.overlay.id || ''));
  } else if (found.node && Array.isArray(found.node.nodeOverlays)) {
    found.node.nodeOverlays = found.node.nodeOverlays.filter((o) => String(o?.id || '') !== String(found.overlay.id || ''));
  }
  scheduleCalibSave();
  setSelection(null);
  renderCalibBoxes();
}

function selectOwnerNodeFromOverlaySelection() {
  const structure = getStructure();
  const sel = calibState.selection;
  if (!structure || !sel || sel.type !== 'overlay') return;
  if (sel.scope !== 'node' || !sel.nodeId) return;
  const node = findNode(structure, sel.nodeId);
  if (!node) return;
  setSelection({ type: 'node', id: String(node.id || '') });
}

async function segmentSelectedOverlay() {
  const task = calibState.currentTask;
  const structure = getStructure();
  const sel = calibState.selection;
  if (!task || !structure || !sel || sel.type !== 'overlay') return;
  const found = findOverlayBySelection(structure, sel);
  if (!found) return;

  const ov = found.overlay;
  try {
    setCalibStatus(t('calibSegmenting'), 'working');
    const response = await fetch('/api/vision/overlays/resolve-one', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: task.image,
        imageWidth: task.image?.width || 0,
        imageHeight: task.image?.height || 0,
        textItems: Array.isArray(task.meta?.textItems) ? task.meta.textItems : [],
        overlay: {
          id: ov.id,
          bbox: ov.bbox,
          kind: ov.kind,
          granularity: ov.granularity,
          fgPoints: Array.isArray(ov.fgPoints) ? ov.fgPoints : [],
          bgPoints: Array.isArray(ov.bgPoints) ? ov.bgPoints : []
        },
        overlayOptions: { tightenBbox: Boolean(state.overlayTrimEnabled) }
      })
    });

    if (!response.ok) {
      const msg = await readErrorMessage(response);
      throw new Error(msg || 'Overlay resolve failed.');
    }

    const data = await response.json();
    const out = data && typeof data.overlay === 'object' ? data.overlay : null;
    if (!out) throw new Error('Invalid overlay response.');
    const ok = Boolean(out.ok);
    if (!ok) {
      const reason = String(out.reason || 'failed');
      const detail = String(out.detail || '');
      throw new Error(detail ? `${reason}: ${detail}` : reason);
    }
    pushCalibUndo('segmentOverlay');
	    if (typeof out.dataUrl === 'string') ov.dataUrl = out.dataUrl;
	    if (out.bbox && typeof out.bbox === 'object') {
	      const bb = out.bbox;
	      ov.bbox = clampBoxToImage(bb);
	    }
	    scheduleCalibSave();
	    renderCalibBoxes();
	    renderCalibPoints();
    syncInspectorFromSelection();
    setCalibStatus(t('calibSegmented'), 'success');
  } catch (err) {
    reportClientError(err, { phase: 'calibration.segment' });
    setCalibStatus(t('calibSegmentFailed', { error: err.message || String(err) }), 'error');
  }
}

async function applyCalibrationToCanvas() {
  const task = calibState.currentTask;
  const structure = getStructure();
  if (!task || !structure) return;
  try {
    setCalibStatus(t('calibApplying'), 'working');
    setStatus(aiStatus, t('structureRendering'), 'working');
    const rendered = await structureToXml(structure, task.image);
    let xml = typeof rendered === 'string' ? rendered : rendered.xml;
    xml = sanitizeMxGraphXml(xml);
    xml = autoLayoutMxGraphXml(xml, { repositionVertices: false });
    xml = sanitizeMxGraphXml(xml);
    const validation = validateMxGraphXml(xml);
    if (!validation.ok) throw new Error(validation.error);
    await loadXmlWithGuard(xml);

    state.lastStructure = structure;
    state.lastImages = [task.image];
    state.lastPrompt = String(task.prompt || promptInput.value.trim() || '');
    state.lastProviderType = getSelectedProvider()?.type || '';
    state.currentJson = null;

    renderOverlayFailures(task.meta);
    const backendTag = task?.meta?.backend ? ` (${String(task.meta.backend)})` : '';
    setCalibStatus(t('calibApplied'), 'success');
    setStatus(aiStatus, `${t('diagramLoaded')}${backendTag}`, 'success');
    closeCalibration();

    // Optional critic pass: compare reference vs rendered, then offer to apply improvements.
    if (state.criticEnabled && task?.image?.dataUrl) {
      try {
        const providerType = state.lastProviderType || getSelectedProvider()?.type || '';
        setStatus(aiStatus, t('criticReviewing'), 'working');
        const renderedPng = await requestExportPng();
        const out = await fetchCritic([task.image], providerType, state.lastPrompt, renderedPng);
        const nextStructure = out && typeof out.structure === 'object' ? out.structure : null;
        const nextMeta = out && typeof out.meta === 'object' ? out.meta : null;
        if (!nextStructure) throw new Error('Invalid critic response');

        const shouldApply = window.confirm(t('criticApplyConfirm'));
        if (!shouldApply) {
          setStatus(aiStatus, `${t('diagramLoaded')}${backendTag}`, 'success');
          return;
        }

        setStatus(aiStatus, t('criticApplying'), 'working');
        const rendered2 = await structureToXml(nextStructure, task.image);
        let xml2 = typeof rendered2 === 'string' ? rendered2 : rendered2.xml;
        xml2 = sanitizeMxGraphXml(xml2);
        xml2 = autoLayoutMxGraphXml(xml2, { repositionVertices: false });
        xml2 = sanitizeMxGraphXml(xml2);
        const v2 = validateMxGraphXml(xml2);
        if (!v2.ok) throw new Error(v2.error);
        await loadXmlWithGuard(xml2);

        state.lastStructure = nextStructure;
        renderOverlayFailures(nextMeta);
        const backendTag2 = nextMeta?.backend ? ` (${String(nextMeta.backend)})` : '';

        task.structure = nextStructure;
        task.meta = nextMeta;
        task.updatedAt = Date.now();
        await putCalibTask(task);

        setStatus(aiStatus, `${t('diagramLoaded')}${backendTag2}`, 'success');
      } catch (err) {
        reportClientError(err, { phase: 'critic.afterApply' });
        setStatus(aiStatus, t('criticFailed', { error: err.message || String(err) }), 'error');
      }
    }
  } catch (err) {
    reportClientError(err, { phase: 'calibration.apply' });
    setCalibStatus(t('calibApplyFailed', { error: err.message || String(err) }), 'error');
    setStatus(aiStatus, t('generationFailed', { error: err.message || String(err) }), 'error');
  }
}

function applyI18n() {
  try {
    document.title = t('brandTitle');
  } catch (err) {
    // ignore
  }
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    el.textContent = t(key);
  });

  document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
    const key = el.getAttribute('data-i18n-placeholder');
    el.setAttribute('placeholder', t(key));
  });

  document.querySelectorAll('[data-i18n-title]').forEach((el) => {
    const key = el.getAttribute('data-i18n-title');
    el.setAttribute('title', t(key));
    el.setAttribute('aria-label', t(key));
  });

  setStatus(aiStatus, t('idle'), 'idle');
  setStatus(importStatus, t('idle'), 'idle');
  renderProviderSelect();
  renderImagePreviews();
  renderCalibShapeOptions();
  renderCalibHistory(calibState.tasks);
  if (calibState.open) syncInspectorFromSelection();
}

function setStatus(el, text, variant = 'idle') {
  el.textContent = text;
  el.dataset.state = variant;
}

function formatOverlayFailureStats(stats) {
  if (!stats || typeof stats !== 'object') return '';
  const parts = [];
  if (stats.mode) parts.push(String(stats.mode));
  const ratio = Number(stats.maskRatio);
  if (Number.isFinite(ratio)) parts.push(`ratio=${ratio.toFixed(3)}`);
  const iou = Number(stats.iou);
  if (Number.isFinite(iou)) parts.push(`iou=${iou.toFixed(2)}`);
  const area = Number(stats.maskArea);
  if (Number.isFinite(area)) parts.push(`area=${Math.round(area)}`);
  if (stats.device) parts.push(`dev=${String(stats.device)}`);
  return parts.join(' ');
}

const OVERLAY_KIND_LABEL_KEY = {
  icon: 'overlayKindIcon',
  photo: 'overlayKindPhoto',
  chart: 'overlayKindChart',
  plot: 'overlayKindPlot',
  '3d': 'overlayKind3d',
  noise: 'overlayKindNoise',
  screenshot: 'overlayKindScreenshot'
};

function overlayKindLabel(kind) {
  const k = String(kind || '').toLowerCase();
  const key = OVERLAY_KIND_LABEL_KEY[k];
  return key ? t(key) : k;
}

function granularityLabel(granularity) {
  const g = String(granularity || '').trim();
  if (g === 'alphaMask') return t('granularityAlphaMask');
  if (g === 'opaqueRect') return t('granularityOpaqueRect');
  return g;
}

const OVERLAY_FAILURE_REASON_LABELS = {
  sam2_unavailable: { en: 'sam2_unavailable', zh: 'SAM2 不可用' },
  sam2_predict_failed: { en: 'sam2_predict_failed', zh: 'SAM2 分割失败' },
  sam2_no_mask: { en: 'sam2_no_mask', zh: 'SAM2 无结果' },
  no_foreground: { en: 'no_foreground', zh: '无前景' },
  too_small: { en: 'too_small', zh: '太小' },
  near_empty: { en: 'near_empty', zh: '近乎空白' },
  texty: { en: 'texty', zh: '疑似文字' },
  ignored: { en: 'ignored', zh: '已忽略' },
  exception: { en: 'exception', zh: '异常' }
};

function overlayFailureReasonLabel(reason) {
  const r = String(reason || '');
  const entry = OVERLAY_FAILURE_REASON_LABELS[r];
  if (entry && entry[state.language]) return entry[state.language];
  return r || (state.language === 'zh' ? '失败' : 'failed');
}

function renderOverlayFailures(meta) {
  if (!overlayFailuresPanel || !overlayFailuresList) return;
  const failures = meta && Array.isArray(meta.overlayFailures) ? meta.overlayFailures : [];
  overlayFailuresList.innerHTML = '';

  if (!failures.length) {
    overlayFailuresPanel.dataset.hidden = 'true';
    if (retryOverlaysBtn) retryOverlaysBtn.disabled = true;
    return;
  }

  overlayFailuresPanel.dataset.hidden = 'false';
  if (retryOverlaysBtn) retryOverlaysBtn.disabled = false;

  const maxItems = 60;
  failures.slice(0, maxItems).forEach((f) => {
    const id = f && typeof f.id === 'string' ? f.id : '';
    const kind = f && typeof f.kind === 'string' ? f.kind : '';
    const granularity = f && typeof f.granularity === 'string' ? f.granularity : '';
    const reason = f && typeof f.reason === 'string' ? f.reason : 'failed';
    const detail = f && typeof f.detail === 'string' ? f.detail : '';
    const bbox = f && f.bbox && typeof f.bbox === 'object' ? f.bbox : null;

    const x = bbox ? Math.round(Number(bbox.x) || 0) : 0;
    const y = bbox ? Math.round(Number(bbox.y) || 0) : 0;
    const w = bbox ? Math.round(Number(bbox.w ?? bbox.width) || 0) : 0;
    const h = bbox ? Math.round(Number(bbox.h ?? bbox.height) || 0) : 0;
    const bboxText = bbox ? `${t('bbox')}:${x},${y},${w}x${h}` : '';
    const statsText = formatOverlayFailureStats(f && typeof f.stats === 'object' ? f.stats : null);
    const metaText = [
      kind ? `${t('overlayKind')}:${overlayKindLabel(kind)}` : '',
      granularity ? `${t('overlayGranularity')}:${granularityLabel(granularity)}` : '',
      bboxText,
      statsText
    ]
      .filter(Boolean)
      .join(' ');

    const item = document.createElement('div');
    item.className = 'overlay-failure-item';

    const line1 = document.createElement('div');
    line1.className = 'line1';

    const left = document.createElement('div');
    left.textContent = id || 'overlay';

    const badge = document.createElement('div');
    badge.className = 'badge';
    badge.textContent = overlayFailureReasonLabel(reason);

    line1.appendChild(left);
    line1.appendChild(badge);

    const line2 = document.createElement('div');
    line2.className = 'line2';
    line2.textContent = metaText || '';

    item.appendChild(line1);
    item.appendChild(line2);

    if (detail) {
      const line3 = document.createElement('div');
      line3.className = 'line3';
      line3.textContent = String(detail).slice(0, 240);
      item.appendChild(line3);
    }
    overlayFailuresList.appendChild(item);
  });
}

function truncateText(value, maxLen = 2000) {
  const text = String(value || '');
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen)}…`;
}

async function reportClientError(error, context = {}) {
  try {
    const payload = {
      name: error?.name || 'Error',
      message: truncateText(error?.message || String(error || '')),
      stack: truncateText(error?.stack || ''),
      context,
      userAgent: navigator.userAgent
    };
    await fetch(CLIENT_ERROR_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (err) {
    // ignore
  }
}

function escapeXml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function encodeDataUrlForMxStyle(url) {
  const s = String(url || '');
  if (!s.startsWith('data:')) return s;
  // mxGraph style strings are ';' delimited; data URLs contain ';base64'.
  // draw.io/mxGraph accepts %3B in data: URLs.
  return s.replace(/;/g, '%3B');
}

function parseStyle(style) {
  const map = {};
  if (!style) return map;
  style.split(';').forEach((item) => {
    if (!item) return;
    const parts = item.split('=');
    const key = parts[0];
    const value = parts.length > 1 ? parts.slice(1).join('=') : '1';
    if (key) {
      map[key] = value;
    }
  });
  return map;
}

function shapeToStyle(shape) {
  return shapeStyleMap[shape] || shapeStyleMap.rectangle;
}

function styleToShape(style) {
  const map = parseStyle(style);
  if (map.shape && styleShapeMap[map.shape]) {
    return styleShapeMap[map.shape];
  }
  return 'rectangle';
}

function handleToStyle(handle, kind) {
  if (!handle || !handlePositions[handle]) {
    return '';
  }
  const pos = handlePositions[handle];
  const prefix = kind === 'source' ? 'exit' : 'entry';
  return `${prefix}X=${pos.x};${prefix}Y=${pos.y};${prefix}Dx=0;${prefix}Dy=0;`;
}

function styleToHandle(style, kind) {
  const map = parseStyle(style);
  const prefix = kind === 'source' ? 'exit' : 'entry';
  const x = map[`${prefix}X`];
  const y = map[`${prefix}Y`];
  if (x === undefined || y === undefined) return null;
  const numX = parseFloat(x);
  const numY = parseFloat(y);
  if (numX === 0 && numY === 0.5) return 'left';
  if (numX === 1 && numY === 0.5) return 'right';
  if (numX === 0.5 && numY === 0) return 'top';
  if (numX === 0.5 && numY === 1) return 'bottom';
  return null;
}

function jsonToXml(flow) {
  const nodes = Array.isArray(flow.nodes) ? flow.nodes : [];
  const edges = Array.isArray(flow.edges) ? flow.edges : [];

  const usedIds = new Set(['0', '1']);
  const idMap = new Map();
  const makeUniqueId = (prefix) => {
    let i = 1;
    while (usedIds.has(`${prefix}${i}`)) i += 1;
    const id = `${prefix}${i}`;
    usedIds.add(id);
    return id;
  };

  nodes.forEach((node) => {
    const raw = node && node.id != null ? String(node.id) : '';
    if (raw && !usedIds.has(raw) && raw !== '0' && raw !== '1') {
      usedIds.add(raw);
      idMap.set(raw, raw);
      return;
    }
    const safe = makeUniqueId('n');
    idMap.set(raw || safe, safe);
  });

  const lines = [];
  lines.push('<?xml version="1.0" encoding="UTF-8"?>');
  lines.push('<mxfile host="app.diagrams.net" modified="" agent="Research Diagram Studio" version="20.1.0">');
  lines.push('<diagram name="Page-1">');
  lines.push('<mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">');
  lines.push('<root>');
  lines.push('<mxCell id="0"/>');
  lines.push('<mxCell id="1" parent="0"/>');

  nodes.forEach((node) => {
    const raw = node && node.id != null ? String(node.id) : '';
    const id = idMap.get(raw) || makeUniqueId('n');
    const label = node.data && node.data.label ? node.data.label : node.label || '';
    const shape = node.data && node.data.shape ? node.data.shape : 'rectangle';
    const width = node.width || 160;
    const height = node.height || 60;
    const x = node.position ? node.position.x : 0;
    const y = node.position ? node.position.y : 0;
    const style = shapeToStyle(shape);

    lines.push(
      `<mxCell id="${escapeXml(id)}" value="${escapeXml(label)}" style="${escapeXml(style)}" vertex="1" parent="1">`
    );
    lines.push(
      `<mxGeometry x="${x}" y="${y}" width="${width}" height="${height}" as="geometry"/>`
    );
    lines.push('</mxCell>');
  });

  edges.forEach((edge) => {
    const rawEdgeId = edge && edge.id != null ? String(edge.id) : '';
    const id =
      rawEdgeId && !usedIds.has(rawEdgeId) && rawEdgeId !== '0' && rawEdgeId !== '1'
        ? (usedIds.add(rawEdgeId), rawEdgeId)
        : makeUniqueId('e');
    const label = edge.label || '';
    const sourceRaw = edge.source || '';
    const targetRaw = edge.target || '';
    const source = idMap.get(String(sourceRaw)) || String(sourceRaw) || '';
    const target = idMap.get(String(targetRaw)) || String(targetRaw) || '';
    const sourceHandleStyle = handleToStyle(edge.sourceHandle, 'source');
    const targetHandleStyle = handleToStyle(edge.targetHandle, 'target');
    const style = `${edgeBaseStyle}${sourceHandleStyle}${targetHandleStyle}`;

    lines.push(
      `<mxCell id="${escapeXml(id)}" value="${escapeXml(label)}" style="${escapeXml(style)}" edge="1" parent="1" source="${escapeXml(source)}" target="${escapeXml(target)}">`
    );
    lines.push('<mxGeometry relative="1" as="geometry"/>');
    lines.push('</mxCell>');
  });

  lines.push('</root>');
  lines.push('</mxGraphModel>');
  lines.push('</diagram>');
  lines.push('</mxfile>');

  return lines.join('');
}

function xmlToJson(xml) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, 'text/xml');
  const mxCells = Array.from(doc.getElementsByTagName('mxCell'));
  const nodes = [];
  const edges = [];

  mxCells.forEach((cell) => {
    const isVertex = cell.getAttribute('vertex') === '1';
    const isEdge = cell.getAttribute('edge') === '1';
    if (!isVertex && !isEdge) return;

    if (isVertex) {
      const geometry = cell.getElementsByTagName('mxGeometry')[0];
      const x = geometry ? parseFloat(geometry.getAttribute('x') || '0') : 0;
      const y = geometry ? parseFloat(geometry.getAttribute('y') || '0') : 0;
      const width = geometry ? parseFloat(geometry.getAttribute('width') || '160') : 160;
      const height = geometry ? parseFloat(geometry.getAttribute('height') || '60') : 60;
      const label = cell.getAttribute('value') || '';
      const style = cell.getAttribute('style') || '';
      const shape = styleToShape(style);
      nodes.push({
        id: cell.getAttribute('id') || '',
        type: 'flow',
        position: { x, y },
        data: { label, shape },
        width,
        height
      });
    }

    if (isEdge) {
      const style = cell.getAttribute('style') || '';
      const sourceHandle = styleToHandle(style, 'source');
      const targetHandle = styleToHandle(style, 'target');
      const edge = {
        id: cell.getAttribute('id') || '',
        source: cell.getAttribute('source') || '',
        target: cell.getAttribute('target') || '',
        label: cell.getAttribute('value') || '',
        type: 'smoothstep'
      };
      if (sourceHandle) edge.sourceHandle = sourceHandle;
      if (targetHandle) edge.targetHandle = targetHandle;
      edges.push(edge);
    }
  });

  return { nodes, edges };
}

function normalizeXml(data) {
  if (!data) return '';
  const trimmed = data.trim();
  if (trimmed.startsWith('<')) return trimmed;
  try {
    return decodeURIComponent(trimmed);
  } catch (err) {
    return data;
  }
}

function cleanAiResponse(text) {
  if (!text) return '';
  const trimmed = text.trim();
  if (trimmed.startsWith('```')) {
    const lines = trimmed.split('\n');
    lines.shift();
    if (lines[lines.length - 1].startsWith('```')) {
      lines.pop();
    }
    return lines.join('\n').trim();
  }
  return trimmed;
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function validateReactFlowJson(flow) {
  if (!isPlainObject(flow)) {
    return { ok: false, error: 'Top-level JSON must be an object.' };
  }
  if (!Array.isArray(flow.nodes) || flow.nodes.length === 0) {
    return { ok: false, error: '"nodes" must be a non-empty array.' };
  }
  if (!Array.isArray(flow.edges)) {
    return { ok: false, error: '"edges" must be an array.' };
  }

  const nodeIds = new Set();
  for (let i = 0; i < flow.nodes.length; i += 1) {
    const node = flow.nodes[i];
    if (!isPlainObject(node)) {
      return { ok: false, error: `nodes[${i}] must be an object.` };
    }
    const id = node.id != null ? String(node.id) : '';
    if (!id) {
      return { ok: false, error: `nodes[${i}].id is required.` };
    }
    if (id === '0' || id === '1') {
      return { ok: false, error: `nodes[${i}].id must not be "${id}".` };
    }
    if (nodeIds.has(id)) {
      return { ok: false, error: `Duplicate node id "${id}".` };
    }
    nodeIds.add(id);

    const position = node.position;
    if (!isPlainObject(position) || !isFiniteNumber(position.x) || !isFiniteNumber(position.y)) {
      return { ok: false, error: `nodes[${i}].position must contain numeric x/y.` };
    }

    const data = node.data;
    if (!isPlainObject(data) || typeof data.label !== 'string') {
      return { ok: false, error: `nodes[${i}].data.label must be a string.` };
    }

    if (node.width != null && !isFiniteNumber(node.width)) {
      return { ok: false, error: `nodes[${i}].width must be a number.` };
    }
    if (node.height != null && !isFiniteNumber(node.height)) {
      return { ok: false, error: `nodes[${i}].height must be a number.` };
    }
  }

  const edgeIds = new Set();
  for (let i = 0; i < flow.edges.length; i += 1) {
    const edge = flow.edges[i];
    if (!isPlainObject(edge)) {
      return { ok: false, error: `edges[${i}] must be an object.` };
    }
    const id = edge.id != null ? String(edge.id) : '';
    if (!id) {
      return { ok: false, error: `edges[${i}].id is required.` };
    }
    if (edgeIds.has(id)) {
      return { ok: false, error: `Duplicate edge id "${id}".` };
    }
    edgeIds.add(id);

    const source = edge.source != null ? String(edge.source) : '';
    const target = edge.target != null ? String(edge.target) : '';
    if (!source || !target) {
      return { ok: false, error: `edges[${i}] must have "source" and "target".` };
    }
    if (!nodeIds.has(source)) {
      return { ok: false, error: `edges[${i}].source references missing node id "${source}".` };
    }
    if (!nodeIds.has(target)) {
      return { ok: false, error: `edges[${i}].target references missing node id "${target}".` };
    }
  }

  return { ok: true };
}

function validateMxGraphXml(xml) {
  if (typeof xml !== 'string' || !xml.trim()) {
    return { ok: false, error: 'XML is empty.' };
  }
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, 'text/xml');
  const parserErrors = doc.getElementsByTagName('parsererror');
  if (parserErrors.length > 0) {
    return { ok: false, error: 'XML parse error.' };
  }

  const mxfile = doc.getElementsByTagName('mxfile')[0];
  const diagram = doc.getElementsByTagName('diagram')[0];
  const model = doc.getElementsByTagName('mxGraphModel')[0];
  const root = doc.getElementsByTagName('root')[0];
  if (!mxfile || !diagram || !model || !root) {
    return { ok: false, error: 'XML must contain <mxfile><diagram><mxGraphModel><root> structure.' };
  }

  const rootChildren = Array.from(root.childNodes).filter((node) => node.nodeType === 1);
  for (const child of rootChildren) {
    if (child.tagName !== 'mxCell') {
      return { ok: false, error: `<root> must contain only <mxCell> children; found <${child.tagName}>.` };
    }
  }

  const cells = Array.from(doc.getElementsByTagName('mxCell'));
  if (cells.length === 0) {
    return { ok: false, error: 'XML contains no mxCell elements.' };
  }

  const ids = new Set();
  for (const cell of cells) {
    const id = cell.getAttribute('id');
    if (!id) {
      return { ok: false, error: 'All mxCell elements must have an id attribute.' };
    }
    if (ids.has(id)) {
      return { ok: false, error: `Duplicate ID '${id}'` };
    }
    ids.add(id);
  }

  if (!ids.has('0') || !ids.has('1')) {
    return { ok: false, error: 'Missing required root mxCell ids 0 and 1.' };
  }

  const cell0 = cells.find((cell) => cell.getAttribute('id') === '0');
  const cell1 = cells.find((cell) => cell.getAttribute('id') === '1');
  if (!cell0 || !cell1) {
    return { ok: false, error: 'Missing required root mxCell ids 0 and 1.' };
  }
  if ((cell1.getAttribute('parent') || '') !== '0') {
    return { ok: false, error: 'Root mxCell id="1" must have parent="0".' };
  }

  const vertexIds = new Set();
  cells.forEach((cell) => {
    if (cell.getAttribute('vertex') === '1') {
      vertexIds.add(cell.getAttribute('id'));
    }
  });

  // Prevent false "loaded" states: require at least one drawable vertex (not just layers 0/1).
  // Note: layers are usually parent="0" without vertex=1, but this guards empty diagrams.
  if (vertexIds.size === 0) {
    return { ok: false, error: 'XML contains no vertices.' };
  }

  const edgeCells = cells.filter((cell) => cell.getAttribute('edge') === '1');
  for (const edge of edgeCells) {
    const source = edge.getAttribute('source') || '';
    const target = edge.getAttribute('target') || '';
    if (!source || !target) {
      return { ok: false, error: `Edge mxCell '${edge.getAttribute('id')}' must have source and target.` };
    }
    if (!vertexIds.has(source)) {
      return { ok: false, error: `Edge mxCell '${edge.getAttribute('id')}' source references missing vertex '${source}'.` };
    }
    if (!vertexIds.has(target)) {
      return { ok: false, error: `Edge mxCell '${edge.getAttribute('id')}' target references missing vertex '${target}'.` };
    }
  }

  return { ok: true };
}

function mergeStyleString(style, additions) {
  const base = parseStyle(style);
  Object.keys(additions).forEach((key) => {
    if (base[key] === undefined || base[key] === '') {
      base[key] = additions[key];
    }
  });
  return `${Object.entries(base)
    .map(([k, v]) => `${k}=${v}`)
    .join(';')};`;
}

function sanitizeMxGraphXml(xml) {
  if (typeof xml !== 'string' || !xml.trim()) return xml;
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, 'text/xml');
  const parserErrors = doc.getElementsByTagName('parsererror');
  if (parserErrors.length > 0) return xml;

  const model = doc.getElementsByTagName('mxGraphModel')[0];
  const root = doc.getElementsByTagName('root')[0];
  if (!model || !root) return xml;

  // mxGraphModel should only contain <root>. If the model includes stray containers
  // (often <Array> / <Object>) that wrap mxCell nodes, flatten them into <root>.
  const modelChildren = Array.from(model.childNodes).filter((n) => n.nodeType === 1);
  modelChildren.forEach((child) => {
    if (child === root) return;
    if (child.tagName === 'Array' || child.tagName === 'Object') {
      const nestedCells = Array.from(child.getElementsByTagName('mxCell'));
      nestedCells.forEach((c) => {
        try {
          root.appendChild(c);
        } catch (err) {
          // ignore
        }
      });
    }
    try {
      model.removeChild(child);
    } catch (err) {
      // ignore
    }
  });

  const elementChildren = Array.from(root.childNodes).filter((node) => node.nodeType === 1);
  elementChildren.forEach((child) => {
    if (child.tagName !== 'mxCell') {
      // Common invalid output: <Array> ... <mxCell/> ... </Array> under <root>.
      // Flatten its mxCell descendants to salvage the diagram.
      if (child.tagName === 'Array' || child.tagName === 'Object') {
        const nestedCells = Array.from(child.getElementsByTagName('mxCell'));
        nestedCells.forEach((c) => {
          try {
            root.appendChild(c);
          } catch (err) {
            // ignore
          }
        });
      }
      root.removeChild(child);
    }
  });

  const childrenAfter = Array.from(root.childNodes).filter((node) => node.nodeType === 1 && node.tagName === 'mxCell');
  let cell0 = childrenAfter.find((cell) => cell.getAttribute('id') === '0') || null;
  let cell1 = childrenAfter.find((cell) => cell.getAttribute('id') === '1') || null;

  if (!cell0) {
    cell0 = doc.createElement('mxCell');
    cell0.setAttribute('id', '0');
    root.insertBefore(cell0, root.firstChild);
  }

  if (!cell1) {
    cell1 = doc.createElement('mxCell');
    cell1.setAttribute('id', '1');
    cell1.setAttribute('parent', '0');
    root.insertBefore(cell1, cell0.nextSibling);
  } else {
    cell1.setAttribute('parent', '0');
  }

  // If the model mistakenly uses reserved ids 0/1 for vertices/edges, rename them
  // and update edge source/target to match. This prevents import errors like
  // "Duplicate ID '1'".
  const allCells = Array.from(doc.getElementsByTagName('mxCell'));
  const existingIds = new Set(allCells.map((c) => c.getAttribute('id')).filter(Boolean));
  const nextId = buildIdGenerator(existingIds);
  const reservedVertexIdMap = new Map(); // '0'/'1' -> new id

  allCells.forEach((cell) => {
    if (cell === cell0 || cell === cell1) return;
    const id = cell.getAttribute('id');
    const isVertexOrEdge = cell.getAttribute('vertex') === '1' || cell.getAttribute('edge') === '1';
    if ((id === '0' || id === '1') && isVertexOrEdge) {
      const newId = nextId('c_');
      reservedVertexIdMap.set(id, newId);
      cell.setAttribute('id', newId);
    }
  });

  // Basic duplicate/empty id cleanup (best-effort).
  const seen = new Set(['0', '1']);
  allCells.forEach((cell) => {
    if (cell === cell0 || cell === cell1) return;
    let id = cell.getAttribute('id') || '';
    if (!id || seen.has(id)) {
      id = nextId('c_');
      cell.setAttribute('id', id);
    }
    seen.add(id);
  });

  // Update edge endpoints for renamed reserved vertex ids.
  if (reservedVertexIdMap.size > 0) {
    allCells.forEach((cell) => {
      if (cell.getAttribute('edge') !== '1') return;
      const source = cell.getAttribute('source') || '';
      const target = cell.getAttribute('target') || '';
      if (reservedVertexIdMap.has(source)) cell.setAttribute('source', reservedVertexIdMap.get(source));
      if (reservedVertexIdMap.has(target)) cell.setAttribute('target', reservedVertexIdMap.get(target));
    });
  }

  const vertexAdditions = {
    html: '1',
    whiteSpace: 'wrap',
    align: 'center',
    verticalAlign: 'middle',
    rounded: '1',
    strokeColor: '#1a5cff',
    fillColor: '#ffffff',
    fontColor: '#1f1e1b',
    fontSize: '14',
    strokeWidth: '1.5'
  };

  const edgeAdditions = {
    edgeStyle: 'orthogonalEdgeStyle',
    rounded: '1',
    html: '1',
    endArrow: 'block',
    elbow: 'horizontal',
    orthogonalLoop: '1',
    jettySize: 'auto',
    exitX: '1',
    exitY: '0.5',
    entryX: '0',
    entryY: '0.5',
    strokeColor: '#1a5cff',
    strokeWidth: '1.5'
  };

  const isOverlayCell = (cell) => {
    if (!cell) return false;
    const value = (cell.getAttribute('value') || '').trim();
    if (value.startsWith('<img') || value.includes('<img')) return true;
    const parent = cell.getAttribute('parent') || '';
    if (!parent) return false;
    const maybeLayer = allCells.find((c) => c.getAttribute('id') === parent);
    return Boolean(maybeLayer && (maybeLayer.getAttribute('value') || '') === 'Overlays');
  };

  allCells.forEach((cell) => {
    const id = cell.getAttribute('id') || '';
    if (id === '0' || id === '1') return;

    if (cell.getAttribute('vertex') === '1') {
      if (!cell.getAttribute('parent')) {
        cell.setAttribute('parent', '1');
      }
      const style = cell.getAttribute('style') || '';
      if (!style.includes('shape=image') && !isOverlayCell(cell)) {
        cell.setAttribute('style', mergeStyleString(style, vertexAdditions));
      }
      const geom = Array.from(cell.childNodes).find(
        (node) => node.nodeType === 1 && node.tagName === 'mxGeometry' && node.getAttribute('as') === 'geometry'
      );
      if (!geom) {
        const geometry = doc.createElement('mxGeometry');
        geometry.setAttribute('x', '40');
        geometry.setAttribute('y', '40');
        geometry.setAttribute('width', '180');
        geometry.setAttribute('height', '70');
        geometry.setAttribute('as', 'geometry');
        cell.appendChild(geometry);
      }

      // mxCell should not contain other element children at the top level
      // (invalid tags like <Array> can trigger "Could not add object Array").
      Array.from(cell.childNodes)
        .filter((n) => n.nodeType === 1 && n.tagName !== 'mxGeometry')
        .forEach((n) => {
          try {
            cell.removeChild(n);
          } catch (err) {
            // ignore
          }
        });
    }

    if (cell.getAttribute('edge') === '1') {
      if (!cell.getAttribute('parent')) {
        cell.setAttribute('parent', '1');
      }
      const style = cell.getAttribute('style') || '';
      cell.setAttribute('style', mergeStyleString(style, edgeAdditions));
      const geom = Array.from(cell.childNodes).find(
        (node) => node.nodeType === 1 && node.tagName === 'mxGeometry' && node.getAttribute('as') === 'geometry'
      );
      if (!geom) {
        const geometry = doc.createElement('mxGeometry');
        geometry.setAttribute('relative', '1');
        geometry.setAttribute('as', 'geometry');
        cell.appendChild(geometry);
      } else {
        geom.setAttribute('relative', '1');
        // Remove any explicit waypoints/points so orthogonal routing can cleanly re-route.
        Array.from(geom.childNodes)
          .filter((n) => n.nodeType === 1)
          .forEach((n) => {
            try {
              geom.removeChild(n);
            } catch (err) {
              // ignore
            }
          });
      }

      // mxCell top-level children other than mxGeometry are invalid for edges too.
      Array.from(cell.childNodes)
        .filter((n) => n.nodeType === 1 && n.tagName !== 'mxGeometry')
        .forEach((n) => {
          try {
            cell.removeChild(n);
          } catch (err) {
            // ignore
          }
        });
    }
  });

  const serializer = new XMLSerializer();
  return serializer.serializeToString(doc);
}

function buildIdGenerator(existingIds) {
  const ids = existingIds || new Set();
  return (prefix) => {
    let n = 1;
    let candidate = `${prefix}${n}`;
    while (ids.has(candidate) || ids.has(String(candidate))) {
      n += 1;
      candidate = `${prefix}${n}`;
    }
    ids.add(candidate);
    return candidate;
  };
}

function insertOverlaysIntoXml(xml, overlays, options = {}) {
  if (!overlays || overlays.length === 0) return xml;
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, 'text/xml');
  const parserErrors = doc.getElementsByTagName('parsererror');
  if (parserErrors.length > 0) return xml;

  const root = doc.getElementsByTagName('root')[0];
  if (!root) return xml;

  const cells = Array.from(doc.getElementsByTagName('mxCell'));
  const existing = new Set(cells.map((cell) => cell.getAttribute('id')).filter(Boolean));
  const nextId = buildIdGenerator(existing);

  const imageWidth = Number(options.imageWidth || 0);
  const imageHeight = Number(options.imageHeight || 0);

  const getGeom = (cell) =>
    Array.from(cell.childNodes).find(
      (node) => node.nodeType === 1 && node.tagName === 'mxGeometry' && node.getAttribute('as') === 'geometry'
    );

  const vertexCells = cells.filter((c) => c.getAttribute('vertex') === '1' && !['0', '1'].includes(c.getAttribute('id') || ''));
  const vertexBbox = (() => {
    const points = [];
    vertexCells.forEach((c) => {
      const g = getGeom(c);
      if (!g) return;
      const x = Number(g.getAttribute('x') || 0);
      const y = Number(g.getAttribute('y') || 0);
      const w = Number(g.getAttribute('width') || 0);
      const h = Number(g.getAttribute('height') || 0);
      if (![x, y, w, h].every((v) => Number.isFinite(v))) return;
      points.push({ x, y, w, h });
    });
    if (points.length === 0) return null;
    const minX = Math.min(...points.map((p) => p.x));
    const minY = Math.min(...points.map((p) => p.y));
    const maxX = Math.max(...points.map((p) => p.x + p.w));
    const maxY = Math.max(...points.map((p) => p.y + p.h));
    return { minX, minY, width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY) };
  })();

  const mapFromImageToDiagram = (() => {
    if (!vertexBbox) return null;
    if (!(imageWidth > 0 && imageHeight > 0)) return null;
    const sx = vertexBbox.width / imageWidth;
    const sy = vertexBbox.height / imageHeight;
    if (!Number.isFinite(sx) || !Number.isFinite(sy) || sx <= 0 || sy <= 0) return null;

    // If the diagram already roughly uses the image coordinate system (1:1), just translate origin.
    const nearOne = (v) => v > 0.7 && v < 1.4;
    if (nearOne(sx) && nearOne(sy)) {
      return { s: 1, tx: vertexBbox.minX, ty: vertexBbox.minY };
    }

    const s = Math.min(sx, sy);
    if (!(s > 0.08 && s < 10)) return null;
    const tx = vertexBbox.minX + (vertexBbox.width - imageWidth * s) / 2;
    const ty = vertexBbox.minY + (vertexBbox.height - imageHeight * s) / 2;
    return { s, tx, ty };
  })();

  const desiredLayerValue = String(options.layerValue || 'Overlays');
  let overlayLayer =
    cells.find((cell) => (cell.getAttribute('value') || '') === desiredLayerValue && (cell.getAttribute('parent') || '') === '0') ||
    null;
  if (!overlayLayer) {
    overlayLayer = doc.createElement('mxCell');
    overlayLayer.setAttribute('id', nextId('layer_overlays_'));
    overlayLayer.setAttribute('value', desiredLayerValue);
    overlayLayer.setAttribute('parent', '0');
    root.appendChild(overlayLayer);
  }
  const layerId = overlayLayer.getAttribute('id');

  overlays.forEach((overlay) => {
    if (!overlay || !overlay.dataUrl || !overlay.geometry) return;
    let { x, y, width, height } = overlay.geometry;
    if (![x, y, width, height].every((v) => typeof v === 'number' && Number.isFinite(v))) return;

    if (mapFromImageToDiagram) {
      const { s, tx, ty } = mapFromImageToDiagram;
      x = x * s + tx;
      y = y * s + ty;
      width = width * s;
      height = height * s;
    }

    const cell = doc.createElement('mxCell');
    cell.setAttribute('id', nextId('ov_'));
    // Use HTML label <img> to avoid mxGraph style-string ';' delimiters in data URLs.
    cell.setAttribute('value', `<img src="${String(overlay.dataUrl)}" style="width:100%;height:100%;display:block;" />`);
    cell.setAttribute('vertex', '1');
    cell.setAttribute('parent', layerId);

    const style = [
      'html=1',
      'shape=label',
      'overflow=fill',
      'rounded=0',
      'strokeColor=none',
      'fillColor=none',
      'shadow=0'
    ].join(';');
    cell.setAttribute('style', `${style};`);

    const geom = doc.createElement('mxGeometry');
    geom.setAttribute('x', String(Math.round(x)));
    geom.setAttribute('y', String(Math.round(y)));
    geom.setAttribute('width', String(Math.max(1, Math.round(width))));
    geom.setAttribute('height', String(Math.max(1, Math.round(height))));
    geom.setAttribute('as', 'geometry');
    cell.appendChild(geom);

    root.appendChild(cell);
  });

  const serializer = new XMLSerializer();
  return serializer.serializeToString(doc);
}

function autoLayoutReactFlowJson(flow) {
  if (!flow || !Array.isArray(flow.nodes) || flow.nodes.length === 0) return flow;
  const nodes = flow.nodes.map((n) => ({ ...n }));
  const edges = Array.isArray(flow.edges) ? flow.edges : [];

  const nodeById = new Map(nodes.map((n) => [String(n.id), n]));
  const indegree = new Map(nodes.map((n) => [String(n.id), 0]));
  const out = new Map(nodes.map((n) => [String(n.id), []]));

  edges.forEach((e) => {
    const s = String(e.source || '');
    const t = String(e.target || '');
    if (!nodeById.has(s) || !nodeById.has(t)) return;
    out.get(s).push(t);
    indegree.set(t, (indegree.get(t) || 0) + 1);
  });

  const layer = new Map(nodes.map((n) => [String(n.id), 0]));
  const queue = [];
  indegree.forEach((d, id) => {
    if (d === 0) queue.push(id);
  });

  const processed = new Set();
  while (queue.length > 0) {
    const id = queue.shift();
    processed.add(id);
    const base = layer.get(id) || 0;
    out.get(id).forEach((to) => {
      layer.set(to, Math.max(layer.get(to) || 0, base + 1));
      indegree.set(to, (indegree.get(to) || 0) - 1);
      if ((indegree.get(to) || 0) === 0) {
        queue.push(to);
      }
    });
  }

  // Break cycles if any remain.
  nodes.forEach((n) => {
    const id = String(n.id);
    if (processed.has(id)) return;
    // place after its predecessors if possible
    let best = 0;
    edges.forEach((e) => {
      if (String(e.target) === id) {
        best = Math.max(best, (layer.get(String(e.source)) || 0) + 1);
      }
    });
    layer.set(id, best);
  });

  const layers = new Map();
  nodes.forEach((n) => {
    const id = String(n.id);
    const l = layer.get(id) || 0;
    if (!layers.has(l)) layers.set(l, []);
    layers.get(l).push(id);
  });

  // Simple barycenter ordering
  const positions = new Map();
  Array.from(layers.keys())
    .sort((a, b) => a - b)
    .forEach((l) => {
      layers.get(l).forEach((id, idx) => positions.set(id, idx));
    });

  const maxIter = 3;
  const sortedLayers = Array.from(layers.keys()).sort((a, b) => a - b);
  for (let iter = 0; iter < maxIter; iter += 1) {
    // forward
    for (let i = 1; i < sortedLayers.length; i += 1) {
      const l = sortedLayers[i];
      const ids = layers.get(l);
      const scored = ids.map((id) => {
        const preds = edges
          .filter((e) => String(e.target) === id)
          .map((e) => String(e.source))
          .filter((p) => positions.has(p));
        const avg = preds.length ? preds.reduce((s, p) => s + (positions.get(p) || 0), 0) / preds.length : positions.get(id) || 0;
        return { id, avg };
      });
      scored.sort((a, b) => a.avg - b.avg);
      layers.set(
        l,
        scored.map((s) => s.id)
      );
      layers.get(l).forEach((id, idx) => positions.set(id, idx));
    }
    // backward
    for (let i = sortedLayers.length - 2; i >= 0; i -= 1) {
      const l = sortedLayers[i];
      const ids = layers.get(l);
      const scored = ids.map((id) => {
        const succs = out.get(id) || [];
        const avg = succs.length ? succs.reduce((s, t) => s + (positions.get(t) || 0), 0) / succs.length : positions.get(id) || 0;
        return { id, avg };
      });
      scored.sort((a, b) => a.avg - b.avg);
      layers.set(
        l,
        scored.map((s) => s.id)
      );
      layers.get(l).forEach((id, idx) => positions.set(id, idx));
    }
  }

  // Place left-to-right columns
  const marginX = 60;
  const marginY = 60;
  const colGap = 260;
  const rowGap = 140;

  sortedLayers.forEach((l) => {
    const ids = layers.get(l) || [];
    ids.forEach((id, idx) => {
      const node = nodeById.get(id);
      if (!node) return;
      node.position = { x: marginX + l * colGap, y: marginY + idx * rowGap };
    });
  });

  return { ...flow, nodes };
}

function autoLayoutMxGraphXml(xml, options = {}) {
  if (typeof xml !== 'string' || !xml.trim()) return xml;
  const repositionVertices = options.repositionVertices !== false;
  const force = options.force === true;
  const tuneEdges = options.tuneEdges === undefined ? repositionVertices : options.tuneEdges !== false;
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, 'text/xml');
  const parserErrors = doc.getElementsByTagName('parsererror');
  if (parserErrors.length > 0) return xml;

  const root = doc.getElementsByTagName('root')[0];
  if (!root) return xml;

  const cells = Array.from(doc.getElementsByTagName('mxCell'));
  const vertices = cells.filter((c) => c.getAttribute('vertex') === '1');
  const edges = cells.filter((c) => c.getAttribute('edge') === '1');

  const overlayLayers = new Set(
    cells
      .filter((c) => {
        const value = (c.getAttribute('value') || '').trim();
        const parent = (c.getAttribute('parent') || '').trim();
        if (parent !== '0') return false;
        return value === 'Overlays' || value === 'OverlayImages' || value === 'OverlayLabels';
      })
      .map((c) => c.getAttribute('id'))
      .filter(Boolean)
  );

  const isOverlay = (cell) => {
    if (!cell) return false;
    const style = cell.getAttribute('style') || '';
    if (style.includes('shape=image')) return true;
    const value = (cell.getAttribute('value') || '').trim();
    if (value.startsWith('<img') || value.includes('<img')) return true;
    const parent = cell.getAttribute('parent') || '';
    return overlayLayers.has(parent);
  };

  // Exclude overlays and non-top-level children from layout.
  const layoutVertices = vertices.filter((c) => (c.getAttribute('parent') || '') === '1' && !isOverlay(c));
  const vertexById = new Map(layoutVertices.map((c) => [String(c.getAttribute('id') || ''), c]));

  const overlayVertices = vertices.filter((c) => isOverlay(c));
  const getGeom = (cell) =>
    Array.from(cell.childNodes).find(
      (node) => node.nodeType === 1 && node.tagName === 'mxGeometry' && node.getAttribute('as') === 'geometry'
    );

  const bboxOf = (cellsToMeasure) => {
    const points = [];
    cellsToMeasure.forEach((c) => {
      const g = getGeom(c);
      if (!g) return;
      const x = Number(g.getAttribute('x') || 0);
      const y = Number(g.getAttribute('y') || 0);
      const w = Number(g.getAttribute('width') || 0);
      const h = Number(g.getAttribute('height') || 0);
      if (![x, y, w, h].every((v) => Number.isFinite(v))) return;
      points.push({ x, y, w, h });
    });
    if (points.length === 0) return null;
    const minX = Math.min(...points.map((p) => p.x));
    const minY = Math.min(...points.map((p) => p.y));
    const maxX = Math.max(...points.map((p) => p.x + p.w));
    const maxY = Math.max(...points.map((p) => p.y + p.h));
    const width = Math.max(1, maxX - minX);
    const height = Math.max(1, maxY - minY);
    return { minX, minY, maxX, maxY, width, height };
  };

  const canReposition = (() => {
    if (!repositionVertices) return false;
    if (force) return true;
    if (vertexById.size < 3) return false;
    const coords = [];
    layoutVertices.forEach((cell) => {
      const geom = getGeom(cell);
      if (!geom) return;
      const x = Number(geom.getAttribute('x') || 0);
      const y = Number(geom.getAttribute('y') || 0);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      coords.push({ x, y });
    });
    if (coords.length < 3) return false;
    const xs = coords.map((c) => c.x);
    const ys = coords.map((c) => c.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const bboxW = maxX - minX;
    const bboxH = maxY - minY;
    const unique = new Set(coords.map((c) => `${Math.round(c.x)}:${Math.round(c.y)}`)).size;
    const looksStacked = unique <= Math.ceil(coords.length * 0.5);
    const looksTiny = bboxW < 260 && bboxH < 220;
    return looksStacked || looksTiny;
  })();

  const indegree = new Map(Array.from(vertexById.keys()).map((id) => [id, 0]));
  const out = new Map(Array.from(vertexById.keys()).map((id) => [id, []]));

  edges.forEach((e) => {
    const s = String(e.getAttribute('source') || '');
    const t = String(e.getAttribute('target') || '');
    if (!vertexById.has(s) || !vertexById.has(t)) return;
    out.get(s).push(t);
    indegree.set(t, (indegree.get(t) || 0) + 1);
  });

  const layer = new Map(Array.from(vertexById.keys()).map((id) => [id, 0]));
  const queue = [];
  indegree.forEach((d, id) => {
    if (d === 0) queue.push(id);
  });

  const processed = new Set();
  while (queue.length > 0) {
    const id = queue.shift();
    processed.add(id);
    const base = layer.get(id) || 0;
    out.get(id).forEach((to) => {
      layer.set(to, Math.max(layer.get(to) || 0, base + 1));
      indegree.set(to, (indegree.get(to) || 0) - 1);
      if ((indegree.get(to) || 0) === 0) queue.push(to);
    });
  }

  // Break cycles if any remain.
  Array.from(vertexById.keys()).forEach((id) => {
    if (processed.has(id)) return;
    let best = 0;
    edges.forEach((e) => {
      if (String(e.getAttribute('target') || '') === id) {
        best = Math.max(best, (layer.get(String(e.getAttribute('source') || '')) || 0) + 1);
      }
    });
    layer.set(id, best);
  });

  const layers = new Map();
  Array.from(vertexById.keys()).forEach((id) => {
    const l = layer.get(id) || 0;
    if (!layers.has(l)) layers.set(l, []);
    layers.get(l).push(id);
  });

  const positions = new Map();
  const sortedLayers = Array.from(layers.keys()).sort((a, b) => a - b);
  sortedLayers.forEach((l) => {
    layers.get(l).sort();
    layers.get(l).forEach((id, idx) => positions.set(id, idx));
  });

  const maxIter = 3;
  for (let iter = 0; iter < maxIter; iter += 1) {
    for (let i = 1; i < sortedLayers.length; i += 1) {
      const l = sortedLayers[i];
      const ids = layers.get(l);
      const scored = ids.map((id) => {
        const preds = edges
          .filter((e) => String(e.getAttribute('target') || '') === id)
          .map((e) => String(e.getAttribute('source') || ''))
          .filter((p) => positions.has(p));
        const avg = preds.length ? preds.reduce((s, p) => s + (positions.get(p) || 0), 0) / preds.length : positions.get(id) || 0;
        return { id, avg };
      });
      scored.sort((a, b) => a.avg - b.avg);
      layers.set(
        l,
        scored.map((s) => s.id)
      );
      layers.get(l).forEach((id, idx) => positions.set(id, idx));
    }

    for (let i = sortedLayers.length - 2; i >= 0; i -= 1) {
      const l = sortedLayers[i];
      const ids = layers.get(l);
      const scored = ids.map((id) => {
        const succs = out.get(id) || [];
        const avg = succs.length ? succs.reduce((s, t) => s + (positions.get(t) || 0), 0) / succs.length : positions.get(id) || 0;
        return { id, avg };
      });
      scored.sort((a, b) => a.avg - b.avg);
      layers.set(
        l,
        scored.map((s) => s.id)
      );
      layers.get(l).forEach((id, idx) => positions.set(id, idx));
    }
  }

  const marginX = 80;
  const marginY = 80;
  const colGapMin = 160;
  const rowGapMin = 110;

  if (canReposition) {
    const before = bboxOf(layoutVertices);

    // Compute per-layer max width/height for spacing.
    const layerDims = new Map();
    sortedLayers.forEach((l) => {
      const ids = layers.get(l) || [];
      let maxW = 0;
      let maxH = 0;
      ids.forEach((id) => {
        const cell = vertexById.get(id);
        const geom = cell ? getGeom(cell) : null;
        if (!geom) return;
        const w = Number(geom.getAttribute('width') || 0);
        const h = Number(geom.getAttribute('height') || 0);
        if (Number.isFinite(w)) maxW = Math.max(maxW, w);
        if (Number.isFinite(h)) maxH = Math.max(maxH, h);
      });
      layerDims.set(l, { maxW: Math.max(80, maxW), maxH: Math.max(40, maxH) });
    });

    // X positions are cumulative by layer width.
    let xCursor = marginX;
    sortedLayers.forEach((l) => {
      const ids = layers.get(l) || [];
      const dim = layerDims.get(l) || { maxW: 180, maxH: 70 };
      let yCursor = marginY;
      ids.forEach((id) => {
        const cell = vertexById.get(id);
        if (!cell) return;
        const geom = getGeom(cell);
        if (!geom) return;
        const w = Number(geom.getAttribute('width') || dim.maxW) || dim.maxW;
        const h = Number(geom.getAttribute('height') || dim.maxH) || dim.maxH;
        geom.setAttribute('x', String(Math.round(xCursor + (dim.maxW - w) / 2)));
        geom.setAttribute('y', String(Math.round(yCursor)));
        yCursor += h + rowGapMin;
      });
      xCursor += dim.maxW + colGapMin;
    });

    // If overlays exist, transform them with the same affine mapping derived from vertex bbox changes.
    const after = bboxOf(layoutVertices);
    if (before && after && overlayVertices.length > 0) {
      const sx = after.width / Math.max(1, before.width);
      const sy = after.height / Math.max(1, before.height);
      const s = Number.isFinite(sx) && Number.isFinite(sy) ? Math.min(sx, sy) : 1;
      const tx = after.minX - before.minX * s;
      const ty = after.minY - before.minY * s;

      // Only apply if the scaling is reasonable; otherwise just translate by delta.
      const scaleOk = s > 0.25 && s < 4.0;
      const dx = after.minX - before.minX;
      const dy = after.minY - before.minY;

      overlayVertices.forEach((cell) => {
        const geom = getGeom(cell);
        if (!geom) return;
        const x = Number(geom.getAttribute('x') || 0);
        const y = Number(geom.getAttribute('y') || 0);
        const w = Number(geom.getAttribute('width') || 0);
        const h = Number(geom.getAttribute('height') || 0);
        if (![x, y, w, h].every((v) => Number.isFinite(v))) return;
        if (scaleOk) {
          geom.setAttribute('x', String(Math.round(x * s + tx)));
          geom.setAttribute('y', String(Math.round(y * s + ty)));
          geom.setAttribute('width', String(Math.max(1, Math.round(w * s))));
          geom.setAttribute('height', String(Math.max(1, Math.round(h * s))));
        } else {
          geom.setAttribute('x', String(Math.round(x + dx)));
          geom.setAttribute('y', String(Math.round(y + dy)));
        }
      });
    }
  }

  // Encourage clean routing only when we also reposition vertices (otherwise preserve the image-derived routing intent).
  if (tuneEdges) {
    edges.forEach((e) => {
      const style = e.getAttribute('style') || '';
      const additions = {
        edgeStyle: 'orthogonalEdgeStyle',
        rounded: '1',
        html: '1',
        endArrow: 'block',
        elbow: 'horizontal',
        orthogonalLoop: '1',
        jettySize: 'auto',
        exitX: '1',
        exitY: '0.5',
        entryX: '0',
        entryY: '0.5',
        exitDx: '0',
        exitDy: '0',
        entryDx: '0',
        entryDy: '0'
      };
      e.setAttribute('style', mergeStyleString(style, additions));
    });
  }

  const serializer = new XMLSerializer();
  return serializer.serializeToString(doc);
}

function buildRepairPrompt(originalPrompt, format, previousOutput, error) {
  const target = format === 'json' ? 'React Flow JSON' : 'draw.io mxGraph XML';
  return [
    'You are fixing your previous diagram output so it can be imported into diagrams.net without errors.',
    `Original request:\n${originalPrompt}`,
    `Validation error:\n${error}`,
    `Previous output:\n${previousOutput}`,
    `Fix the output and return ONLY valid ${target}.`,
    'Do not include explanations, markdown, or code fences.'
  ].join('\n\n');
}

function formatProviderWarning(warningCode, warningText) {
  const text = String(warningText || '').trim();
  const base =
    warningCode === 'images_not_supported'
      ? t('warningImagesNotSupported')
      : warningCode === 'images_rejected'
        ? t('warningImagesRejected')
        : '';

  if (!text) return base;
  if (!base) return text;

  const match = text.match(/\((.+)\)\s*$/);
  if (match && match[1]) {
    return `${base} (${match[1]})`;
  }

  return base;
}

function getSelectedProviderId() {
  return localStorage.getItem(SELECTED_PROVIDER_KEY);
}

function setSelectedProviderId(id) {
  localStorage.setItem(SELECTED_PROVIDER_KEY, id);
}

async function fetchServerProviders() {
  try {
    const response = await fetch('/api/providers');
    if (!response.ok) return;
    const data = await response.json();
    const list = Array.isArray(data.providers) ? data.providers : [];
    state.providersServer = {};
    list.forEach((provider) => {
      if (provider && provider.type) {
        state.providersServer[provider.type] = provider;
      }
    });
    state.primaryProvider = data.primary || state.primaryProvider || '';
    renderProviderSelect();
  } catch (err) {
    // ignore
  }
}

function renderProviderSelect() {
  const selected = state.primaryProvider || getSelectedProviderId();
  providerSelect.innerHTML = '';

  providerCatalog.forEach((provider) => {
    const option = document.createElement('option');
    option.value = provider.type;
    option.textContent = provider.labelKey ? t(provider.labelKey) : provider.label;
    providerSelect.appendChild(option);
  });

  if (selected && providerCatalogByType[selected]) {
    providerSelect.value = selected;
  } else if (selected) {
    localStorage.removeItem(SELECTED_PROVIDER_KEY);
  }

  if (!providerSelect.value && providerSelect.options.length > 0) {
    providerSelect.selectedIndex = 0;
    setSelectedProviderId(providerSelect.value);
  }

  updateProviderStatus();
  updateProviderFields();
}

function toggleKeyVisibility() {
  if (!providerKey) return;
  const isHidden = providerKey.type === 'password';
  providerKey.type = isHidden ? 'text' : 'password';
  toggleKeyVisibilityBtn.classList.toggle('is-visible', isHidden);
}

async function saveProvider() {
  const selected = getSelectedProvider();
  if (!selected) {
    setStatus(aiStatus, t('selectProviderFirst'), 'error');
    return;
  }

  const apiKeyTrimmed = providerKey.value.trim();

  const payload = {
    type: selected.type,
    model: providerModel.value.trim(),
    imageModel: providerImageModel ? providerImageModel.value.trim() : '',
    apiKey: apiKeyTrimmed,
    baseUrl: providerBase.value.trim(),
    primary: selected.type
  };

  if (!payload.model) {
    setStatus(aiStatus, t('providerMissing'), 'error');
    return;
  }

  if (selected.requiresBase && !payload.baseUrl) {
    setStatus(aiStatus, t('providerBaseRequired'), 'error');
    return;
  }

  try {
    const response = await fetch('/api/providers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.error || 'Save failed');
    }

    const data = await response.json();
    const list = Array.isArray(data.providers) ? data.providers : [];
    state.providersServer = {};
    list.forEach((provider) => {
      if (provider && provider.type) {
        state.providersServer[provider.type] = provider;
      }
    });
    state.primaryProvider = data.primary || state.primaryProvider;
    renderProviderSelect();
    updateProviderFields();
    if (apiKeyTrimmed) {
      providerKeyCache[selected.type] = apiKeyTrimmed;
      persistProviderKeyCache();
    }
    setStatus(aiStatus, t('providerSaved'), 'success');
  } catch (err) {
    setStatus(aiStatus, t('providerSaveFailed', { error: err.message }), 'error');
  }
}

function getSelectedProvider() {
  const type = providerSelect.value || state.primaryProvider || providerCatalog[0]?.type;
  if (!type) return null;
  const catalog = providerCatalogByType[type];
  if (!catalog) return null;
  const config = state.providersServer[type] || {};
  return { ...catalog, ...config, type };
}

function postToEditor(message) {
  if (!iframe.contentWindow) return;
  iframe.contentWindow.postMessage(JSON.stringify(message), '*');
}

function loadXml(xml) {
  state.currentXml = xml;
  postToEditor({ action: 'load', xml, autosave: 1 });
}

function loadXmlWithGuard(xml) {
  state.currentXml = xml;
  postToEditor({ action: 'load', xml, autosave: 1 });

  if (!state.iframeReady) {
    return Promise.resolve();
  }

  if (state.pendingCanvasLoad) {
    state.pendingCanvasLoad.resolve();
    state.pendingCanvasLoad = null;
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      const err = new Error('Canvas load timed out');
      state.pendingCanvasLoad = null;
      reject(err);
    }, 8000);

    state.pendingCanvasLoad = {
      resolve: () => {
        clearTimeout(timeout);
        state.pendingCanvasLoad = null;
        resolve();
      },
      reject: (err) => {
        clearTimeout(timeout);
        state.pendingCanvasLoad = null;
        reject(err);
      }
    };
  });
}

function requestExportXml() {
  if (!state.iframeReady) {
    return Promise.reject(new Error('Editor not ready'));
  }

  if (state.pendingExport) {
    return Promise.reject(new Error('Export already running'));
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      state.pendingExport = null;
      reject(new Error('Export timed out'));
    }, 10000);

    state.pendingExport = {
      format: 'xml',
      resolve: (xml) => {
        clearTimeout(timeout);
        resolve(xml);
      },
      reject
    };

    postToEditor({ action: 'export', format: 'xml', spin: t('editorExporting') });
  });
}

function requestExportPng() {
  if (!state.iframeReady) {
    return Promise.reject(new Error('Editor not ready'));
  }

  if (state.pendingExport) {
    return Promise.reject(new Error('Export already running'));
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      state.pendingExport = null;
      reject(new Error('Export timed out'));
    }, 20000);

    state.pendingExport = {
      format: 'png',
      resolve: (dataUrl) => {
        clearTimeout(timeout);
        resolve(dataUrl);
      },
      reject
    };

    postToEditor({ action: 'export', format: 'png', spin: t('editorRendering') });
  });
}

function downloadFile(name, content) {
  const blob = new Blob([content], { type: 'text/plain' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

async function handleImportJson(text) {
  try {
    const flow = JSON.parse(text);
    const xml = jsonToXml(flow);
    state.currentJson = flow;
    loadXml(xml);
    setStatus(importStatus, t('jsonImported'), 'success');
  } catch (err) {
    setStatus(importStatus, t('jsonImportFailed', { error: err.message }), 'error');
  }
}

async function handleImportXml(text) {
  try {
    loadXml(text);
    setStatus(importStatus, t('xmlImported'), 'success');
  } catch (err) {
    setStatus(importStatus, t('xmlImportFailed', { error: err.message }), 'error');
  }
}

async function syncFromCanvas() {
  try {
    setStatus(importStatus, t('syncingFromCanvas'), 'working');
    const xml = await requestExportXml();
    const normalized = normalizeXml(xml);
    state.currentXml = normalized;
    state.currentJson = xmlToJson(normalized);
    setStatus(importStatus, t('canvasSynced'), 'success');
  } catch (err) {
    setStatus(importStatus, t('syncFailed', { error: err.message }), 'error');
  }
}

async function handleExportXml() {
  try {
    setStatus(importStatus, t('exportingXml'), 'working');
    const xml = await requestExportXml();
    const normalized = normalizeXml(xml);
    downloadFile('diagram.xml', normalized);
    setStatus(importStatus, t('xmlExported'), 'success');
  } catch (err) {
    setStatus(importStatus, t('exportFailed', { error: err.message }), 'error');
  }
}

async function handleExportJson() {
  try {
    setStatus(importStatus, t('exportingJson'), 'working');
    const xml = await requestExportXml();
    const normalized = normalizeXml(xml);
    const json = xmlToJson(normalized);
    downloadFile('diagram.json', JSON.stringify(json, null, 2));
    setStatus(importStatus, t('jsonExported'), 'success');
  } catch (err) {
    setStatus(importStatus, t('exportFailed', { error: err.message }), 'error');
  }
}

async function readImages() {
  const files = Array.from(imageInput.files || []);
  const result = [];
  // Higher cap keeps small icons/text crisp for paper-figure screenshots (better OCR/SAM2/LLM).
  // If a provider has strict limits, we still downscale progressively.
  const targetMaxBytes = 6_500_000;

  function estimateDataUrlBytes(dataUrl) {
    const base64 = String(dataUrl || '').split(',')[1] || '';
    return Math.floor((base64.length * 3) / 4);
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new Error('File read failed'));
      reader.readAsDataURL(file);
    });
  }

  async function rasterizeToDataUrl(file, maxDim, format, quality) {
    const bitmap = await createImageBitmap(file);
    const meta = { origWidth: bitmap.width, origHeight: bitmap.height };
    const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { alpha: false });
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(bitmap, 0, 0, width, height);
    if (typeof bitmap.close === 'function') {
      bitmap.close();
    }

    const kind = String(format || '').toLowerCase() === 'png' ? 'png' : 'jpeg';
    const mimeOut = kind === 'png' ? 'image/png' : 'image/jpeg';
    const dataUrl = kind === 'png' ? canvas.toDataURL('image/png') : canvas.toDataURL('image/jpeg', quality);
    return { dataUrl, type: mimeOut, width, height, ...meta };
  }

  async function preprocessImage(file) {
    const mime = String(file.type || '').toLowerCase();
    const allowed = new Set(['image/png', 'image/jpeg', 'image/webp']);
    if (!allowed.has(mime)) {
      throw new Error(t('imageUnsupportedType', { type: mime || file.name || 'unknown' }));
    }

    // Best-quality path: use the original bytes (no canvas resample / no re-encode) when small enough.
    try {
      const bitmap = await createImageBitmap(file);
      const bw = Number(bitmap.width) || 0;
      const bh = Number(bitmap.height) || 0;
      const directUrl = await readFileAsDataUrl(file);
      const directBytes = estimateDataUrlBytes(directUrl);
      const maxDim = Math.max(bw, bh);
      if (directUrl.startsWith('data:image/') && directBytes > 0 && directBytes <= targetMaxBytes && maxDim <= 2800) {
        if (typeof bitmap.close === 'function') bitmap.close();
        return { name: file.name, dataUrl: directUrl, type: mime, width: bw, height: bh, origWidth: bw, origHeight: bh };
      }
      if (typeof bitmap.close === 'function') bitmap.close();
    } catch (err) {
      // Fall back to rasterization below.
    }

    // Always try PNG first (avoids re-encoding JPEG artifacts and preserves thin lines/text).
    const pngVariants = [{ maxDim: 2400 }, { maxDim: 2048 }, { maxDim: 1600 }, { maxDim: 1280 }, { maxDim: 1024 }];
    for (const variant of pngVariants) {
      try {
        const out = await rasterizeToDataUrl(file, variant.maxDim, 'png', 0.95);
        if (estimateDataUrlBytes(out.dataUrl) <= targetMaxBytes) {
          return { name: file.name, dataUrl: out.dataUrl, type: out.type, width: out.width, height: out.height, origWidth: out.origWidth, origHeight: out.origHeight };
        }
      } catch (err) {
        // fall back below
        break;
      }
    }

    const variants = [
      { maxDim: 1600, quality: 0.9 },
      { maxDim: 1280, quality: 0.85 },
      { maxDim: 1024, quality: 0.8 },
      { maxDim: 896, quality: 0.75 }
    ];

    let last = null;
    let lastMeta = null;
    for (const variant of variants) {
      try {
        const out = await rasterizeToDataUrl(file, variant.maxDim, 'jpeg', variant.quality);
        last = out.dataUrl;
        lastMeta = { type: out.type, width: out.width, height: out.height, origWidth: out.origWidth, origHeight: out.origHeight };
        if (estimateDataUrlBytes(out.dataUrl) <= targetMaxBytes) {
          return { name: file.name, dataUrl: out.dataUrl, ...lastMeta };
        }
      } catch (err) {
        // fall back below
        break;
      }
    }

    if (last) {
      return { name: file.name, dataUrl: last, ...(lastMeta || {}) };
    }

    throw new Error(t('imageDecodeFailed', { name: file.name || 'unknown' }));
  }

  for (const file of files) {
    const processed = await preprocessImage(file);
    result.push(processed);
  }
  return result;
}

async function readErrorMessage(response) {
  if (!response) return '';
  const status = Number(response.status);
  const statusText = String(response.statusText || '').trim();
  const contentType = String(response.headers?.get?.('content-type') || '');
  const prefix = Number.isFinite(status) && status ? `HTTP ${status}${statusText ? ` ${statusText}` : ''}` : 'HTTP error';

  try {
    if (contentType.includes('application/json')) {
      const body = await response.json().catch(() => ({}));
      const obj = body && typeof body === 'object' ? body : {};
      const msg = String(obj.error || obj.message || obj.detail || '').trim();
      const details = [];
      if (obj.code) details.push(`code=${String(obj.code).trim()}`);
      if (obj.phase) details.push(`phase=${String(obj.phase).trim()}`);
      if (Array.isArray(obj.missingFields) && obj.missingFields.length) {
        details.push(`missing=${obj.missingFields.map((v) => String(v)).join(',')}`);
      }
      const suffix = details.length ? ` (${details.join(', ')})` : '';
      return msg ? `${prefix}: ${msg}${suffix}` : `${prefix}${suffix}`;
    }
    const text = await response.text();
    const trimmed = String(text || '').trim().slice(0, 240);
    return trimmed ? `${prefix}: ${trimmed}` : prefix;
  } catch (err) {
    return prefix;
  }
}

function overlayCacheKey(images, providerType) {
  const first = images && images[0] ? `${images[0].name}|${images[0].type}|${images[0].width}x${images[0].height}` : '';
  return `${providerType || ''}::${first}`;
}

async function detectOverlays(images, providerType) {
  if (!images || images.length === 0) return [];
  const first = images[0];
  if (!first.dataUrl) return [];

  const key = overlayCacheKey(images, providerType);
  if (state.overlaysCache[key]) {
    return state.overlaysCache[key];
  }

  setStatus(aiStatus, t('overlayDetecting'), 'working');
  const response = await fetch('/api/overlays/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      providerType,
      images: [first],
      imageWidth: first.width || 0,
      imageHeight: first.height || 0
    })
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || 'Overlay detection failed');
  }

  const data = await response.json();
  const overlays = Array.isArray(data.overlays) ? data.overlays : [];
  state.overlaysCache[key] = overlays;
  return overlays;
}

async function fetchStructure(images, providerType, prompt) {
  const first = images && images[0] ? images[0] : null;
  if (!first) throw new Error('No image');
  async function postStructure(imageObj) {
    const response = await fetch('/api/vision/structure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...(providerType ? { providerType } : {}),
        prompt: String(prompt || ''),
        images: [imageObj],
        imageWidth: imageObj.width || 0,
        imageHeight: imageObj.height || 0,
        qualityMode: state.qualityMode,
        overlayOptions: { tightenBbox: Boolean(state.overlayTrimEnabled) }
      })
    });

    if (!response.ok) {
      const msg = await readErrorMessage(response);
      return { ok: false, status: response.status, error: msg || 'Structure extraction failed' };
    }

    const data = await response.json();
    if (!data || !data.structure || typeof data.structure !== 'object') {
      return { ok: false, status: 200, error: 'Invalid structure response' };
    }
    return { ok: true, status: 200, data: { structure: data.structure, meta: data.meta || null } };
  }

  function shouldRetryWithReencode(status, message) {
    const msg = String(message || '').toLowerCase();
    if (!msg) return false;
    if (status === 413) return true;
    if (msg.includes('unable to process input image')) return true;
    if (msg.includes('cannot process') && msg.includes('image')) return true;
    if (msg.includes('invalid') && msg.includes('image')) return true;
    if (msg.includes('payload') && msg.includes('too large')) return true;
    if (msg.includes('request entity too large')) return true;
    return false;
  }

  function shouldRetryTransient(status, message) {
    const s = Number(status);
    const msg = String(message || '').toLowerCase();
    if (Number.isFinite(s) && (s === 403 || s === 429 || s === 500 || s === 502 || s === 503 || s === 504)) return true;
    if (msg.includes('gateway timeout')) return true;
    if (msg.includes('timeout')) return true;
    if (msg.includes('temporarily unavailable')) return true;
    return false;
  }

	  async function delay(ms) {
	    return new Promise((r) => setTimeout(r, ms));
	  }
	
	  const firstAttempt = await postStructure(first);
	  if (firstAttempt.ok) return { ...firstAttempt.data, image: first };
	
	  if (shouldRetryTransient(firstAttempt.status, firstAttempt.error)) {
	    await delay(700);
	    const retryAttempt = await postStructure(first);
	    if (retryAttempt.ok) return { ...retryAttempt.data, image: first };
	  }
	
	  if (first.dataUrl && shouldRetryWithReencode(firstAttempt.status, firstAttempt.error)) {
	    try {
	      const alt = await reencodeImageForVision(first, { maxDim: 1024, prefer: 'png' });
	      const secondAttempt = await postStructure(alt);
	      if (secondAttempt.ok) return { ...secondAttempt.data, image: alt };
	
	      const alt2 = await reencodeImageForVision(first, { maxDim: 896, prefer: 'jpeg', quality: 0.75 });
	      const thirdAttempt = await postStructure(alt2);
	      if (thirdAttempt.ok) return { ...thirdAttempt.data, image: alt2 };
	    } catch (err) {
	      // ignore, fall through
	    }
	  }

	  throw new Error(firstAttempt.error || 'Structure extraction failed');
	}

function rescaleStructureCoordinates(structure, fromSize, toSize) {
  const fromW = Number(fromSize?.width || 0);
  const fromH = Number(fromSize?.height || 0);
  const toW = Number(toSize?.width || 0);
  const toH = Number(toSize?.height || 0);
  if (!(fromW > 0 && fromH > 0 && toW > 0 && toH > 0)) return structure;
  if (fromW === toW && fromH === toH) return structure;
  if (!structure || typeof structure !== 'object') return structure;

  const sx = toW / fromW;
  const sy = toH / fromH;

  function scaleBox(bb) {
    const b = boxFrom(bb);
    if (!b) return bb;
    return {
      x: Math.round(b.x * sx),
      y: Math.round(b.y * sy),
      w: Math.round(b.w * sx),
      h: Math.round(b.h * sy)
    };
  }

  function scalePoints(list) {
    const pts = Array.isArray(list) ? list : [];
    return pts
      .filter((p) => p && typeof p === 'object')
      .map((p) => ({ x: Math.round(Number(p.x) * sx), y: Math.round(Number(p.y) * sy) }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  }

  const out = cloneForUndo(structure);
  const nodes = Array.isArray(out.nodes) ? out.nodes : [];
  nodes.forEach((n) => {
    if (!n || typeof n !== 'object') return;
    if (n.bbox) n.bbox = scaleBox(n.bbox);
    if (n.textBbox) n.textBbox = scaleBox(n.textBbox);
    if (Array.isArray(n.nodeOverlays)) {
      n.nodeOverlays.forEach((ov) => {
        if (!ov || typeof ov !== 'object') return;
        if (ov.bbox) ov.bbox = scaleBox(ov.bbox);
        if (Array.isArray(ov.fgPoints)) ov.fgPoints = scalePoints(ov.fgPoints);
        if (Array.isArray(ov.bgPoints)) ov.bgPoints = scalePoints(ov.bgPoints);
      });
    }
    if (Array.isArray(n.innerShapes)) {
      n.innerShapes.forEach((s) => {
        if (!s || typeof s !== 'object') return;
        if (s.bbox) s.bbox = scaleBox(s.bbox);
      });
    }
  });

  if (Array.isArray(out.overlays)) {
    out.overlays.forEach((ov) => {
      if (!ov || typeof ov !== 'object') return;
      if (ov.bbox) ov.bbox = scaleBox(ov.bbox);
      if (Array.isArray(ov.fgPoints)) ov.fgPoints = scalePoints(ov.fgPoints);
      if (Array.isArray(ov.bgPoints)) ov.bgPoints = scalePoints(ov.bgPoints);
    });
  }

  return out;
}

async function fetchCritic(images, providerType, prompt, rendered) {
  const first = images && images[0] ? images[0] : null;
  if (!first) throw new Error('No image');

  async function postCritic(imageObj) {
    const response = await fetch('/api/vision/critic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...(providerType ? { providerType } : {}),
        prompt: String(prompt || ''),
        rendered: String(rendered || ''),
        images: [imageObj],
        imageWidth: imageObj.width || 0,
        imageHeight: imageObj.height || 0,
        qualityMode: state.qualityMode,
        overlayOptions: { tightenBbox: Boolean(state.overlayTrimEnabled) }
      })
    });

    if (!response.ok) {
      const msg = await readErrorMessage(response);
      return { ok: false, status: response.status, error: msg || 'Critic failed' };
    }

    const data = await response.json();
    if (!data || !data.structure || typeof data.structure !== 'object') {
      return { ok: false, status: 200, error: 'Invalid critic response' };
    }
    return { ok: true, status: 200, data: { structure: data.structure, meta: data.meta || null } };
  }

  function shouldRetryTransient(status, message) {
    const s = Number(status);
    const msg = String(message || '').toLowerCase();
    if (Number.isFinite(s) && (s === 403 || s === 429 || s === 500 || s === 502 || s === 503 || s === 504)) return true;
    if (msg.includes('gateway timeout')) return true;
    if (msg.includes('timeout')) return true;
    if (msg.includes('temporarily unavailable')) return true;
    return false;
  }

  async function delay(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  const attempt1 = await postCritic(first);
  if (attempt1.ok) return attempt1.data;

  if (shouldRetryTransient(attempt1.status, attempt1.error)) {
    await delay(900);
    const attempt2 = await postCritic(first);
    if (attempt2.ok) return attempt2.data;
  }

  throw new Error(attempt1.error || 'Critic failed');
}

async function fetchVisionDebug(images, prompt) {
  const first = images && images[0] ? images[0] : null;
  if (!first) throw new Error('No image');
  const response = await fetch('/api/vision/debug/annotate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: String(prompt || ''),
      images: [first],
      imageWidth: first.width || 0,
      imageHeight: first.height || 0
    })
  });
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || 'Vision debug failed');
  }
  return await response.json();
}

const bitmapCache = new Map();

async function getBitmapFromDataUrl(dataUrl) {
  const key = String(dataUrl || '').slice(0, 128) + String(dataUrl || '').length;
  if (bitmapCache.has(key)) return bitmapCache.get(key);
  const blob = await fetch(dataUrl).then((r) => r.blob());
  const bmp = await createImageBitmap(blob);
  bitmapCache.set(key, bmp);
  return bmp;
}

async function reencodeImageForVision(imageObj, options) {
  const prefer = String(options?.prefer || 'jpeg').toLowerCase();
  const kind = prefer === 'png' ? 'png' : 'jpeg';
  const maxDim = Math.max(128, Number(options?.maxDim) || 1024);
  const quality = Number(options?.quality);
  const q = Number.isFinite(quality) ? Math.max(0.5, Math.min(0.95, quality)) : 0.82;

  const bmp = await getBitmapFromDataUrl(imageObj.dataUrl);
  const scale = Math.min(1, maxDim / Math.max(bmp.width, bmp.height));
  const width = Math.max(1, Math.round(bmp.width * scale));
  const height = Math.max(1, Math.round(bmp.height * scale));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { alpha: false });
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(bmp, 0, 0, width, height);

  const type = kind === 'png' ? 'image/png' : 'image/jpeg';
  const dataUrl = kind === 'png' ? canvas.toDataURL('image/png') : canvas.toDataURL('image/jpeg', q);
  return { ...imageObj, dataUrl, type, width, height };
}

function rgbToHex(r, g, b) {
  const to = (n) => Math.max(0, Math.min(255, Math.round(n || 0))).toString(16).padStart(2, '0');
  return `#${to(r)}${to(g)}${to(b)}`;
}

function luminance(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function dominantColorFromBuckets(buckets) {
  let bestKey = null;
  let bestCount = 0;
  for (const [k, v] of buckets.entries()) {
    if (v > bestCount) {
      bestCount = v;
      bestKey = k;
    }
  }
  if (!bestKey) return null;
  const rBin = (bestKey >> 8) & 0xf;
  const gBin = (bestKey >> 4) & 0xf;
  const bBin = bestKey & 0xf;
  return rgbToHex(rBin * 16 + 8, gBin * 16 + 8, bBin * 16 + 8);
}

async function sampleBoxColors(imageDataUrl, bbox, textBbox) {
  const bmp = await getBitmapFromDataUrl(imageDataUrl);
  const x = Math.max(0, Math.floor(bbox.x));
  const y = Math.max(0, Math.floor(bbox.y));
  const w = Math.max(1, Math.floor(bbox.w ?? bbox.width));
  const h = Math.max(1, Math.floor(bbox.h ?? bbox.height));

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bmp, x, y, w, h, 0, 0, w, h);

  const img = ctx.getImageData(0, 0, w, h);
  const data = img.data;

  const ring = Math.max(2, Math.min(10, Math.round(Math.min(w, h) * 0.08)));
  const step = Math.max(1, Math.round(Math.min(w, h) / 180));

  let textLocal = null;
  if (textBbox && typeof textBbox === 'object') {
    const tx1 = Math.max(0, Math.floor(textBbox.x - x));
    const ty1 = Math.max(0, Math.floor(textBbox.y - y));
    const tx2 = Math.min(w, Math.ceil(textBbox.x + textBbox.w - x));
    const ty2 = Math.min(h, Math.ceil(textBbox.y + textBbox.h - y));
    if (tx2 > tx1 && ty2 > ty1) {
      textLocal = { x1: tx1, y1: ty1, x2: tx2, y2: ty2 };
    }
  }

  const strokeBuckets = new Map();
  const fillBuckets = new Map();

  const isInText = (px, py) => {
    if (!textLocal) return false;
    const pad = 2;
    return px >= textLocal.x1 - pad && px < textLocal.x2 + pad && py >= textLocal.y1 - pad && py < textLocal.y2 + pad;
  };

  for (let py = 0; py < h; py += step) {
    for (let px = 0; px < w; px += step) {
      if (isInText(px, py)) continue;
      const idx = (py * w + px) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const lum = luminance(r, g, b);

      const rBin = r >> 4;
      const gBin = g >> 4;
      const bBin = b >> 4;
      const key = (rBin << 8) | (gBin << 4) | bBin;

      const isStroke = px < ring || px >= w - ring || py < ring || py >= h - ring;
      const buckets = isStroke ? strokeBuckets : fillBuckets;

      // Filter likely background noise for stroke.
      if (isStroke) {
        if (lum > 248) continue;
        if (lum < 8) continue;
      }

      buckets.set(key, (buckets.get(key) || 0) + 1);
    }
  }

  const stroke = dominantColorFromBuckets(strokeBuckets);
  const fill = dominantColorFromBuckets(fillBuckets);
  return { strokeColor: stroke, fillColor: fill };
}

function normalizeShapeId(shapeId) {
  const raw = String(shapeId || '').trim();
  if (!raw) return '';
  const lower = raw.toLowerCase();
  if (lower === 'diamond') return 'rhombus';
  if (lower === 'roundrect' || lower === 'roundrectangle') return '';
  if (lower === 'rect' || lower === 'rectangle') return '';
  return raw;
}

function chooseFontColor(fillColor) {
  const hex = String(fillColor || '');
  if (!hex.startsWith('#') || hex.length !== 7) return '#1f1e1b';
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return luminance(r, g, b) < 135 ? '#ffffff' : '#1f1e1b';
}

function createXmlCell({ id, value, style, parent, vertex, edge, source, target, geometry }) {
  const parts = [];
  parts.push(`<mxCell id="${escapeXml(id)}"`);
  if (value !== undefined) parts.push(` value="${escapeXml(value)}"`);
  if (style) parts.push(` style="${escapeXml(style)}"`);
  if (parent) parts.push(` parent="${escapeXml(parent)}"`);
  if (vertex) parts.push(` vertex="1"`);
  if (edge) parts.push(` edge="1"`);
  if (source) parts.push(` source="${escapeXml(source)}"`);
  if (target) parts.push(` target="${escapeXml(target)}"`);
  parts.push('>');
  if (geometry) {
    const { x, y, width, height, relative } = geometry;
    if (edge) {
      parts.push(`<mxGeometry relative="${relative ? '1' : '1'}" as="geometry"></mxGeometry>`);
    } else {
      parts.push(
        `<mxGeometry x="${Math.round(x)}" y="${Math.round(y)}" width="${Math.max(1, Math.round(width))}" height="${Math.max(1, Math.round(height))}" as="geometry"></mxGeometry>`
      );
    }
  }
  parts.push(`</mxCell>`);
  return parts.join('');
}

async function structureToXml(structure, image) {
  const img = image || {};
  const imgW = Number(img.width || 0);
  const imgH = Number(img.height || 0);
  const marginX = 80;
  const marginY = 80;

  const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
  const edges = Array.isArray(structure.edges) ? structure.edges : [];
  const overlays = Array.isArray(structure.overlays) ? structure.overlays : [];

  if (nodes.length === 0) {
    throw new Error('No nodes detected from image.');
  }

  const overlayImagesLayerId = 'layer_overlay_images';
  const idMap = new Map(); // nodeId -> cellId
  const centerMap = new Map(); // nodeId -> {cx,cy}
  const xmlCells = [];

  // Root cells/layers
  xmlCells.push('<mxCell id="0"></mxCell>');
  xmlCells.push('<mxCell id="1" parent="0"></mxCell>');
  xmlCells.push(`<mxCell id="${overlayImagesLayerId}" value="OverlayImages" parent="0"></mxCell>`);

  // Nodes
  let overlayRequested = 0;
  let overlayInserted = 0;
  for (const node of nodes) {
    if (!node || !node.bbox) continue;
    const id = String(node.id || '');
    const bbox = node.bbox;
    const x = Number(bbox.x) + marginX;
    const y = Number(bbox.y) + marginY;
    const w = Number(bbox.w ?? bbox.width);
    const h = Number(bbox.h ?? bbox.height);
    centerMap.set(id, { cx: x + w / 2, cy: y + h / 2 });

    const cellId = `n_${id}`;
    idMap.set(id, cellId);

    const render = String(node.render || 'shape').toLowerCase();
    if (render === 'text') {
      const text = typeof node.text === 'string' ? node.text : '';
      if (!text) continue;
      xmlCells.push(
        createXmlCell({
          id: cellId,
          value: text,
          style: 'shape=label;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;fillColor=none;strokeColor=none;fontSize=14;',
          parent: '1',
          vertex: true,
          geometry: { x, y, width: w, height: h }
        })
      );
      continue;
    }

    if (render === 'overlay') {
      const dataUrl = typeof node.dataUrl === 'string' ? node.dataUrl : null;
      overlayRequested += 1;
      if (dataUrl && dataUrl.startsWith('data:image/')) overlayInserted += 1;

      const style = dataUrl
        ? `shape=image;imageAspect=0;image=${encodeDataUrlForMxStyle(String(dataUrl))};rounded=0;strokeColor=none;fillColor=none;shadow=0;`
        : 'rounded=1;html=1;whiteSpace=wrap;fillColor=none;strokeColor=#ef4444;dashed=1;strokeWidth=2;';

      xmlCells.push(
        createXmlCell({
          id: cellId,
          value: '',
          style,
          parent: '1',
          vertex: true,
          geometry: { x, y, width: w, height: h }
        })
      );

      const text = typeof node.text === 'string' ? node.text : '';
      if (text) {
        const tb = node.textBbox && typeof node.textBbox === 'object' ? node.textBbox : null;
        const txLocal = tb ? Number(tb.x) - Number(bbox.x) : w * 0.1;
        const tyLocal = tb ? Number(tb.y) - Number(bbox.y) : h * 0.35;
        const tw = tb ? Number(tb.w ?? tb.width) : w * 0.8;
        const th = tb ? Number(tb.h ?? tb.height) : h * 0.3;

        xmlCells.push(
          createXmlCell({
            id: `t_${id}`,
            value: text,
            style: 'shape=label;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;fillColor=none;strokeColor=none;fontSize=14;',
            parent: cellId,
            vertex: true,
            geometry: { x: txLocal, y: tyLocal, width: tw, height: th }
          })
        );
      }

      if (Array.isArray(node.nodeOverlays)) {
        node.nodeOverlays.forEach((ov, idx) => {
          if (!ov) return;
          if (ov.granularity === 'ignore') return;
          overlayRequested += 1;
          const dataUrl2 = ov.dataUrl;
          const bb = ov.bbox || ov.geometry || null;
          if (!dataUrl2 || !bb) return;
          if (typeof dataUrl2 === 'string' && dataUrl2.startsWith('data:image/')) overlayInserted += 1;
          const ox = Number(bb.x) - Number(bbox.x);
          const oy = Number(bb.y) - Number(bbox.y);
          const ow = Number(bb.w ?? bb.width);
          const oh = Number(bb.h ?? bb.height);
          if (![ox, oy, ow, oh].every((v) => Number.isFinite(v))) return;
          xmlCells.push(
            createXmlCell({
              id: `nov_${id}_${idx + 1}`,
              value: '',
              style: `shape=image;imageAspect=0;image=${encodeDataUrlForMxStyle(String(dataUrl2))};rounded=0;strokeColor=none;fillColor=none;shadow=0;`,
              parent: cellId,
              vertex: true,
              geometry: { x: ox, y: oy, width: ow, height: oh }
            })
          );
        });
      }
      continue;
    } else {
      const text = typeof node.text === 'string' ? node.text : '';
      const rawShapeId = String(node.shapeId || '');
      const shapeId = normalizeShapeId(rawShapeId);

      let fillColor = null;
      let strokeColor = null;
      try {
        if (img.dataUrl && imgW > 0 && imgH > 0) {
          const sampled = await sampleBoxColors(img.dataUrl, bbox, node.textBbox);
          fillColor = sampled.fillColor;
          strokeColor = sampled.strokeColor;
        }
      } catch (err) {
        // ignore
      }
      if (node.containerStyle && typeof node.containerStyle === 'object') {
        if (typeof node.containerStyle.fillColor === 'string') fillColor = node.containerStyle.fillColor;
        if (typeof node.containerStyle.strokeColor === 'string') strokeColor = node.containerStyle.strokeColor;
      }
      fillColor = fillColor || '#ffffff';
      strokeColor = strokeColor || '#2563eb';
      const fontColor = chooseFontColor(fillColor);

      const styleParts = [
        'html=1',
        'whiteSpace=wrap',
        'align=center',
        'verticalAlign=middle',
        `strokeColor=${strokeColor}`,
        `fillColor=${fillColor}`,
        `fontColor=${fontColor}`,
        'fontSize=14',
        'strokeWidth=1.5'
      ];
      if (shapeId) {
        styleParts.push(`shape=${shapeId}`);
      }
      const rawLower = rawShapeId.toLowerCase();
      if (rawLower === 'roundrect' || rawLower === 'roundrectangle' || rawLower === 'round rect') {
        styleParts.push('rounded=1');
        styleParts.push('arcSize=18');
      }
      xmlCells.push(
        createXmlCell({
          id: cellId,
          value: '',
          style: `${styleParts.join(';')};`,
          parent: '1',
          vertex: true,
          geometry: { x, y, width: w, height: h }
        })
      );

      // Editable label as child of node container (keeps anchoring when layout changes).
      if (text) {
        const tb = node.textBbox && typeof node.textBbox === 'object' ? node.textBbox : null;
        const txLocal = tb ? Number(tb.x) - Number(bbox.x) : w * 0.1;
        const tyLocal = tb ? Number(tb.y) - Number(bbox.y) : h * 0.35;
        const tw = tb ? Number(tb.w ?? tb.width) : w * 0.8;
        const th = tb ? Number(tb.h ?? tb.height) : h * 0.3;
        xmlCells.push(
          createXmlCell({
            id: `t_${id}`,
            value: text,
            style: 'shape=label;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;fillColor=none;strokeColor=none;fontSize=14;',
            parent: cellId,
            vertex: true,
            geometry: { x: txLocal, y: tyLocal, width: tw, height: th }
          })
        );
      }

      // Inner shapes (treated as layout details inside the node group).
      if (Array.isArray(node.innerShapes)) {
        node.innerShapes.forEach((s, idx) => {
          if (!s) return;
          const bb = s.bbox || s.geometry || null;
          if (!bb) return;
          const sx = Number(bb.x) - Number(bbox.x);
          const sy = Number(bb.y) - Number(bbox.y);
          const sw = Number(bb.w ?? bb.width);
          const sh = Number(bb.h ?? bb.height);
          if (![sx, sy, sw, sh].every((v) => Number.isFinite(v))) return;
          const shapeId2 = normalizeShapeId(s.shapeId || s.shape || 'rectangle');
          const style = s.style || {};
          const fill = typeof style.fillColor === 'string' ? style.fillColor : '#ffffff';
          const stroke = typeof style.strokeColor === 'string' ? style.strokeColor : 'none';
          const gradient = typeof style.gradientColor === 'string' ? style.gradientColor : '';
          const gradDir = typeof style.gradientDirection === 'string' ? style.gradientDirection : '';
          const parts = [
            'html=1',
            'whiteSpace=wrap',
            `fillColor=${fill}`,
            stroke === 'none' ? 'strokeColor=none' : `strokeColor=${stroke}`,
            'rounded=0'
          ];
          if (shapeId2) parts.push(`shape=${shapeId2}`);
          if (gradient) parts.push(`gradientColor=${gradient}`);
          if (gradDir) parts.push(`gradientDirection=${gradDir}`);
          xmlCells.push(
            createXmlCell({
              id: `is_${id}_${idx + 1}`,
              value: '',
              style: `${parts.join(';')};`,
              parent: cellId,
              vertex: true,
              geometry: { x: sx, y: sy, width: sw, height: sh }
            })
          );
        });
      }

      // Overlays inside node (true bitmaps, anchored)
      if (Array.isArray(node.nodeOverlays)) {
        node.nodeOverlays.forEach((ov, idx) => {
          if (!ov) return;
          if (ov.granularity === 'ignore') return;
          overlayRequested += 1;
          const dataUrl = ov.dataUrl;
          const bb = ov.bbox || ov.geometry || null;
          if (!dataUrl || !bb) return;
          if (typeof dataUrl === 'string' && dataUrl.startsWith('data:image/')) overlayInserted += 1;
          const ox = Number(bb.x) - Number(bbox.x);
          const oy = Number(bb.y) - Number(bbox.y);
          const ow = Number(bb.w ?? bb.width);
          const oh = Number(bb.h ?? bb.height);
          if (![ox, oy, ow, oh].every((v) => Number.isFinite(v))) return;
          xmlCells.push(
            createXmlCell({
              id: `nov_${id}_${idx + 1}`,
              value: '',
              style: `shape=image;imageAspect=0;image=${encodeDataUrlForMxStyle(String(dataUrl))};rounded=0;strokeColor=none;fillColor=none;shadow=0;`,
              parent: cellId,
              vertex: true,
              geometry: { x: ox, y: oy, width: ow, height: oh }
            })
          );
        });
      }
    }
  }

  // Extra overlays (non-node)
  for (let idx = 0; idx < overlays.length; idx += 1) {
    const ov = overlays[idx];
    if (!ov || !ov.bbox) continue;
    if (ov.granularity === 'ignore') continue;
    overlayRequested += 1;
    const dataUrl = typeof ov.dataUrl === 'string' ? ov.dataUrl : null;
    if (!dataUrl || !dataUrl.startsWith('data:image/')) continue;
    overlayInserted += 1;
    const bb = ov.bbox;
    const ox = Number(bb.x) + marginX;
    const oy = Number(bb.y) + marginY;
    const ow = Number(bb.w ?? bb.width);
    const oh = Number(bb.h ?? bb.height);
    xmlCells.push(
      createXmlCell({
        id: `ov_${String(ov.id || idx + 1)}`,
        value: '',
        style: `shape=image;imageAspect=0;image=${encodeDataUrlForMxStyle(String(dataUrl))};rounded=0;strokeColor=none;fillColor=none;shadow=0;`,
        parent: overlayImagesLayerId,
        vertex: true,
        geometry: { x: ox, y: oy, width: ow, height: oh }
      })
    );
  }

  // Edges
  for (const edge of edges) {
    if (!edge) continue;
    const source = idMap.get(String(edge.source || '')) || '';
    const target = idMap.get(String(edge.target || '')) || '';
    if (!source || !target) continue;
    const srcKey = String(edge.source || '');
    const tgtKey = String(edge.target || '');
    const conf = Number(edge.confidence);
    const edgeConf = Number.isFinite(conf) ? conf : 0.6;
    let sSide = String(edge.sourceSide || '').toLowerCase();
    let tSide = String(edge.targetSide || '').toLowerCase();
    const valid = new Set(['left', 'right', 'top', 'bottom']);
    const invalidSides = !valid.has(sSide) || !valid.has(tSide);
    if (edgeConf < 0.7 || invalidSides) {
      const sC = centerMap.get(srcKey);
      const tC = centerMap.get(tgtKey);
      if (sC && tC) {
        const dx = tC.cx - sC.cx;
        const dy = tC.cy - sC.cy;
        if (Math.abs(dx) >= Math.abs(dy)) {
          sSide = dx >= 0 ? 'right' : 'left';
          tSide = dx >= 0 ? 'left' : 'right';
        } else {
          sSide = dy >= 0 ? 'bottom' : 'top';
          tSide = dy >= 0 ? 'top' : 'bottom';
        }
      }
    }
    const sPos = handlePositions[sSide] || handlePositions.right;
    const tPos = handlePositions[tSide] || handlePositions.left;
    const style =
      `${edgeBaseStyle}` +
      `exitX=${sPos.x};exitY=${sPos.y};entryX=${tPos.x};entryY=${tPos.y};` +
      'orthogonalLoop=1;jettySize=auto;endFill=1;strokeColor=#2563eb;strokeWidth=3.5;';
    xmlCells.push(
      createXmlCell({
        id: `e_${String(edge.id || '')}`,
        value: typeof edge.label === 'string' ? edge.label : '',
        style,
        parent: '1',
        edge: true,
        source,
        target,
        geometry: { relative: 1 }
      })
    );
  }

  let xml = `<?xml version="1.0" encoding="UTF-8"?><mxfile><diagram><mxGraphModel><root>${xmlCells.join('')}</root></mxGraphModel></diagram></mxfile>`;
  const dropped = Math.max(0, overlayRequested - overlayInserted);
  // Overlay failures are surfaced via the server's `meta.overlayFailures` panel; avoid treating partial overlay
  // insertion as a hard client error (it is expected when some extractions fail).

  return { xml, overlayStats: { requested: overlayRequested, inserted: overlayInserted, dropped } };
}

function colorDist2(a, b) {
  const dr = a[0] - b[0];
  const dg = a[1] - b[1];
  const db = a[2] - b[2];
  return dr * dr + dg * dg + db * db;
}

function removeBackgroundFromImageData(imageData, options) {
  const { data, width, height } = imageData;
  if (width < 2 || height < 2) return imageData;

  const overrideMean = Array.isArray(options?.mean) && options.mean.length === 3 ? options.mean : null;
  const overrideThreshold = Number(options?.threshold);

  const border = [];
  function pushPixel(x, y) {
    const i = (y * width + x) * 4;
    border.push([data[i], data[i + 1], data[i + 2]]);
  }

  for (let x = 0; x < width; x += 1) {
    pushPixel(x, 0);
    pushPixel(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    pushPixel(0, y);
    pushPixel(width - 1, y);
  }

  const mean = overrideMean
    ? [Number(overrideMean[0]) || 0, Number(overrideMean[1]) || 0, Number(overrideMean[2]) || 0]
    : [0, 0, 0];
  if (!overrideMean) {
    border.forEach((c) => {
      mean[0] += c[0];
      mean[1] += c[1];
      mean[2] += c[2];
    });
    mean[0] /= border.length;
    mean[1] /= border.length;
    mean[2] /= border.length;
  }

  let variance = 0;
  border.forEach((c) => {
    variance += colorDist2(c, mean);
  });
  variance /= Math.max(1, border.length);
  const std = Math.sqrt(variance);
  const threshold = Number.isFinite(overrideThreshold) ? Math.max(10, overrideThreshold) : Math.max(18, std * 2.2);
  const t2 = threshold * threshold;
  const tSoft2 = (threshold * 2.2) * (threshold * 2.2);

  const bg = new Uint8Array(width * height);
  const queue = new Int32Array(width * height);
  let qh = 0;
  let qt = 0;

  function enqueue(x, y) {
    const idx = y * width + x;
    if (bg[idx]) return;
    const i = idx * 4;
    const c = [data[i], data[i + 1], data[i + 2]];
    if (colorDist2(c, mean) <= t2) {
      bg[idx] = 1;
      queue[qt++] = idx;
    }
  }

  for (let x = 0; x < width; x += 1) {
    enqueue(x, 0);
    enqueue(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    enqueue(0, y);
    enqueue(width - 1, y);
  }

  while (qh < qt) {
    const idx = queue[qh++];
    const x = idx % width;
    const y = (idx / width) | 0;

    if (x > 0) enqueue(x - 1, y);
    if (x < width - 1) enqueue(x + 1, y);
    if (y > 0) enqueue(x, y - 1);
    if (y < height - 1) enqueue(x, y + 1);
  }

  for (let i = 0; i < width * height; i += 1) {
    if (bg[i]) {
      data[i * 4 + 3] = 0;
    }
  }

  // Soft edge: reduce alpha for pixels very close to background color at boundaries.
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const idx = y * width + x;
      if (bg[idx]) continue;
      const nearBg =
        bg[idx - 1] ||
        bg[idx + 1] ||
        bg[idx - width] ||
        bg[idx + width] ||
        bg[idx - width - 1] ||
        bg[idx - width + 1] ||
        bg[idx + width - 1] ||
        bg[idx + width + 1];
      if (!nearBg) continue;

      const i = idx * 4;
      const c = [data[i], data[i + 1], data[i + 2]];
      const d2 = colorDist2(c, mean);
      if (d2 <= tSoft2) {
        const a = Math.max(0, Math.min(255, Math.round(((d2 - t2) / Math.max(1, (tSoft2 - t2))) * 255)));
        data[i + 3] = Math.min(data[i + 3], a);
      }
    }
  }

  return imageData;
}

function padBox(box, pad, maxW, maxH) {
  const x = Math.max(0, Math.floor(box.x - pad));
  const y = Math.max(0, Math.floor(box.y - pad));
  const x2 = Math.min(maxW, Math.ceil(box.x + box.width + pad));
  const y2 = Math.min(maxH, Math.ceil(box.y + box.height + pad));
  return { x, y, width: Math.max(1, x2 - x), height: Math.max(1, y2 - y) };
}

function estimateBorderStats(imageData) {
  const { data, width, height } = imageData;
  const border = [];
  const borderKeys = new Map();
  function pushPixel(x, y) {
    const i = (y * width + x) * 4;
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    border.push([r, g, b]);
    const key = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4);
    borderKeys.set(key, (borderKeys.get(key) || 0) + 1);
  }
  for (let x = 0; x < width; x += 1) {
    pushPixel(x, 0);
    pushPixel(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    pushPixel(0, y);
    pushPixel(width - 1, y);
  }
  const mean = [0, 0, 0];
  border.forEach((c) => {
    mean[0] += c[0];
    mean[1] += c[1];
    mean[2] += c[2];
  });
  mean[0] /= Math.max(1, border.length);
  mean[1] /= Math.max(1, border.length);
  mean[2] /= Math.max(1, border.length);

  let variance = 0;
  border.forEach((c) => {
    variance += colorDist2(c, mean);
  });
  variance /= Math.max(1, border.length);
  const std = Math.sqrt(variance);
  const threshold = Math.max(18, std * 2.2);

  // Robust background estimate: dominant quantized border color bucket.
  let bgKey = null;
  let bgCount = 0;
  for (const [k, v] of borderKeys.entries()) {
    if (v > bgCount) {
      bgCount = v;
      bgKey = k;
    }
  }

  let bgMean = mean;
  let bgDominance = border.length > 0 ? bgCount / border.length : 0;
  if (bgKey !== null) {
    const bucket = [];
    border.forEach((c) => {
      const key = ((c[0] >> 4) << 8) | ((c[1] >> 4) << 4) | (c[2] >> 4);
      if (key === bgKey) bucket.push(c);
    });
    const use = bucket.length >= Math.max(10, Math.floor(border.length * 0.15)) ? bucket : border;
    const m = [0, 0, 0];
    use.forEach((c) => {
      m[0] += c[0];
      m[1] += c[1];
      m[2] += c[2];
    });
    m[0] /= Math.max(1, use.length);
    m[1] /= Math.max(1, use.length);
    m[2] /= Math.max(1, use.length);
    bgMean = m;
    bgDominance = border.length > 0 ? bucket.length / border.length : bgDominance;
  }

  let bgVar = 0;
  border.forEach((c) => {
    bgVar += colorDist2(c, bgMean);
  });
  bgVar /= Math.max(1, border.length);
  const bgStd = Math.sqrt(bgVar);
  const bgThreshold = Math.max(14, Math.min(90, bgStd * 2.0 + 18));

  return {
    mean,
    std,
    threshold,
    t2: threshold * threshold,
    bgMean,
    bgStd,
    bgThreshold,
    bgT2: bgThreshold * bgThreshold,
    bgDominance
  };
}

function clampLocalRect(x, y, w, h, maxW, maxH) {
  const xx = Math.max(0, Math.min(Math.floor(x), maxW));
  const yy = Math.max(0, Math.min(Math.floor(y), maxH));
  const x2 = Math.max(xx, Math.min(Math.ceil(x + w), maxW));
  const y2 = Math.max(yy, Math.min(Math.ceil(y + h), maxH));
  return { x: xx, y: yy, w: Math.max(0, x2 - xx), h: Math.max(0, y2 - yy) };
}

function applyMaskBoxes(imageData, padded, maskBoxes) {
  if (!Array.isArray(maskBoxes) || maskBoxes.length === 0) return;
  const { data, width, height } = imageData;
  for (const mb of maskBoxes) {
    if (!mb) continue;
    const local = clampLocalRect(
      Number(mb.x) - padded.x,
      Number(mb.y) - padded.y,
      Number(mb.width ?? mb.w ?? 0),
      Number(mb.height ?? mb.h ?? 0),
      width,
      height
    );
    if (local.w <= 0 || local.h <= 0) continue;
    for (let y = local.y; y < local.y + local.h; y += 1) {
      for (let x = local.x; x < local.x + local.w; x += 1) {
        const i = (y * width + x) * 4;
        data[i + 3] = 0;
      }
    }
  }
}

function findForegroundBounds(imageData, stats, mode) {
  const { data, width, height } = imageData;
  const alphaThresh = 24;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  const bg = stats?.bgMean || stats?.mean || [255, 255, 255];
  const bgT2 = Number(stats?.bgT2) || Number(stats?.t2) || 18 * 18;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      const a = data[i + 3];
      if (a <= alphaThresh) continue;
      let fg = false;
      if (mode === 'alpha') {
        fg = true;
      } else {
        const c = [data[i], data[i + 1], data[i + 2]];
        fg = colorDist2(c, bg) > bgT2;
      }
      if (!fg) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }

  if (maxX < minX || maxY < minY) return null;
  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  return { x: minX, y: minY, w, h };
}

async function extractOverlayDataUrl(imageDataUrl, box) {
  const blob = await fetch(imageDataUrl).then((r) => r.blob());
  const bitmap = await createImageBitmap(blob);
  const maxW = bitmap.width;
  const maxH = bitmap.height;
  const requestedPad = box && Object.prototype.hasOwnProperty.call(box, 'pad') ? Number(box.pad) : null;
  const pad =
    Number.isFinite(requestedPad) && requestedPad >= 0
      ? Math.round(requestedPad)
      : Math.round(Math.max(6, Math.min(32, Math.max(box.width, box.height) * 0.06)));
  const padded = padBox(box, pad, maxW, maxH);

  const canvas = document.createElement('canvas');
  canvas.width = padded.width;
  canvas.height = padded.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true, alpha: true });
  ctx.drawImage(bitmap, padded.x, padded.y, padded.width, padded.height, 0, 0, padded.width, padded.height);
  if (typeof bitmap.close === 'function') bitmap.close();

  const debug = Boolean(box && box.debug);
  const drop = (reason, extra = {}) => {
    if (!debug) return null;
    return { dropped: true, reason: String(reason || 'dropped'), ...extra };
  };

  const imageData = ctx.getImageData(0, 0, padded.width, padded.height);
  const kind = typeof box?.kind === 'string' ? box.kind : '';
  const stats = estimateBorderStats(imageData);

  // Mask out known text areas so overlay crops don't contain text.
  // If the mask is suspiciously huge, ignore it (likely mis-detected text region).
  const maskBoxes = Array.isArray(box?.mask) ? box.mask : [];
  const maskArea = maskBoxes.reduce((acc, mb) => acc + Math.max(0, Number(mb?.width ?? mb?.w ?? 0)) * Math.max(0, Number(mb?.height ?? mb?.h ?? 0)), 0);
  const cropArea = Math.max(1, padded.width * padded.height);
  const useMask = maskBoxes.length > 0 && maskArea / cropArea < 0.6;
  const originalPixels = useMask ? new Uint8ClampedArray(imageData.data) : null;
  if (useMask) {
    applyMaskBoxes(imageData, padded, maskBoxes);
  }

  const removeBgKinds = new Set(['icon', 'equation', 'texture']);
  // For node overlays, remove background only when border is strongly dominated by one color.
  const shouldRemoveBg =
    removeBgKinds.has(kind) || (kind === 'node' && Number(stats.bgDominance) > 0.78 && Number(stats.bgStd) < 55);
  const processed = shouldRemoveBg ? removeBackgroundFromImageData(imageData, { mean: stats.bgMean, threshold: stats.bgThreshold }) : imageData;

  // Tighten crop to actual foreground content.
  const boundsMode = shouldRemoveBg || (Array.isArray(box?.mask) && box.mask.length > 0) ? 'alpha' : 'bg';
  let bounds = findForegroundBounds(processed, stats, boundsMode);
  if (!bounds && useMask && originalPixels) {
    // Mask may have removed the true content; retry without mask.
    processed.data.set(originalPixels);
    bounds = findForegroundBounds(processed, stats, shouldRemoveBg ? 'alpha' : 'bg');
  }
  // If we fail to find a foreground bound (common when border sampling is polluted),
  // fall back to keeping the original crop rather than dropping overlays entirely.
  const fallbackFull = () => ({ x: 0, y: 0, w: padded.width, h: padded.height });
  if (!bounds) {
    bounds = fallbackFull();
  }
  const minKeep = ['icon', 'chart', 'plot'].includes(kind) ? 16 : 24;
  if (bounds.w < minKeep || bounds.h < minKeep) {
    bounds = fallbackFull();
  }

  const pad2 = Math.round(Math.max(2, Math.min(10, Math.min(bounds.w, bounds.h) * 0.05)));
  const bx = Math.max(0, bounds.x - pad2);
  const by = Math.max(0, bounds.y - pad2);
  const bx2 = Math.min(padded.width, bounds.x + bounds.w + pad2);
  const by2 = Math.min(padded.height, bounds.y + bounds.h + pad2);
  const bw = Math.max(1, bx2 - bx);
  const bh = Math.max(1, by2 - by);

  let opaque = 0;
  for (let i = 0; i < processed.data.length; i += 4) {
    if (processed.data[i + 3] > 24) opaque += 1;
  }
  if (opaque < 10 && shouldRemoveBg) {
    // Background removal may have removed too much; fall back to original crop.
    return {
      dataUrl: canvas.toDataURL('image/png'),
      geometry: { x: padded.x, y: padded.y, width: padded.width, height: padded.height }
    };
  }
  if (opaque < 10) return drop('near_empty', { opaque, bounds });

  // Filter: discard "texty" crops (many tiny components, no dominant shape).
  // Keep icons/photos/3d more permissively.
  if (shouldRemoveBg && opaque > 200 && !['icon', 'photo', '3d', 'chart', 'plot', 'node'].includes(kind)) {
    const width = padded.width;
    const height = padded.height;
    const mask = new Uint8Array(width * height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = y * width + x;
        mask[idx] = processed.data[idx * 4 + 3] > 24 ? 1 : 0;
      }
    }

    const visited = new Uint8Array(width * height);
    let components = 0;
    let largest = 0;
    const stack = [];
    const push = (x, y) => {
      stack.push(x, y);
    };

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = y * width + x;
        if (!mask[idx] || visited[idx]) continue;
        components += 1;
        let size = 0;
        visited[idx] = 1;
        push(x, y);
        while (stack.length) {
          const cy = stack.pop();
          const cx = stack.pop();
          size += 1;
          if (cx > 0) {
            const n = cy * width + (cx - 1);
            if (mask[n] && !visited[n]) {
              visited[n] = 1;
              push(cx - 1, cy);
            }
          }
          if (cx < width - 1) {
            const n = cy * width + (cx + 1);
            if (mask[n] && !visited[n]) {
              visited[n] = 1;
              push(cx + 1, cy);
            }
          }
          if (cy > 0) {
            const n = (cy - 1) * width + cx;
            if (mask[n] && !visited[n]) {
              visited[n] = 1;
              push(cx, cy - 1);
            }
          }
          if (cy < height - 1) {
            const n = (cy + 1) * width + cx;
            if (mask[n] && !visited[n]) {
              visited[n] = 1;
              push(cx, cy + 1);
            }
          }
        }
        if (size > largest) largest = size;
        if (components >= 40) break;
      }
      if (components >= 40) break;
    }

    const dominance = opaque > 0 ? largest / opaque : 1;
    if (components >= 18 && dominance < 0.12) {
      return drop('texty', { components, dominance });
    }
  }

  ctx.putImageData(processed, 0, 0);

  const outCanvas = document.createElement('canvas');
  outCanvas.width = bw;
  outCanvas.height = bh;
  const outCtx = outCanvas.getContext('2d', { alpha: true });
  outCtx.drawImage(canvas, bx, by, bw, bh, 0, 0, bw, bh);

  return {
    dataUrl: outCanvas.toDataURL('image/png'),
    geometry: { x: padded.x + bx, y: padded.y + by, width: bw, height: bh }
  };
}

async function generateReferenceImage() {
  const selected = getSelectedProvider();
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setStatus(aiStatus, t('enterPromptFirst'), 'error');
    return;
  }

  if (!state.imageModelEnabled) {
    setStatus(aiStatus, t('imageModelDisabled'), 'error');
    openSettings();
    return;
  }

  if (!selected) {
    setStatus(aiStatus, t('selectProviderFirst'), 'error');
    return;
  }

  if (!String(selected.imageModel || '').trim()) {
    setStatus(aiStatus, t('imageModelMissing'), 'error');
    openSettings();
    return;
  }

  if (!isProviderConfigured(selected)) {
    setStatus(aiStatus, t('providerNotConfigured'), 'error');
    openSettings();
    return;
  }

  try {
    let images = [];
    if ((imageInput.files || []).length > 0) {
      setStatus(aiStatus, t('imagePreprocessing'), 'working');
      images = await readImages();
    }

    setStatus(aiStatus, t('generatingRefImage'), 'working');
    const response = await fetch('/api/image/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        providerType: selected.type,
        prompt,
        images,
        allowImageModel: true
      })
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || 'Image generation failed');
    }

    const image = data && typeof data.image === 'object' ? data.image : {};
    const dataUrl = String(image.dataUrl || '').trim();
    if (!dataUrl.startsWith('data:image/')) {
      throw new Error('Image generation returned an invalid image.');
    }

    const file = await dataUrlToFile(dataUrl, image.name || 'generated.png');
    const existing = Array.from(imageInput.files || []);
    const next = [file, ...existing].slice(0, 4);
    updateImageInputFiles(next);
    renderImagePreviews(t('refImageGenerated'));

    const xmlRadio = document.querySelector('input[name="format"][value="xml"]');
    if (xmlRadio) xmlRadio.checked = true;

    await openCalibrationFromCurrentInput();
  } catch (err) {
    reportClientError(err, { phase: 'image.generate', providerType: selected.type });
    setStatus(aiStatus, t('refImageFailed', { error: err.message || String(err) }), 'error');
  }
}

async function generateDiagram() {
  const selected = getSelectedProvider();
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setStatus(aiStatus, t('enterPromptFirst'), 'error');
    return;
  }

  const format = document.querySelector('input[name="format"]:checked').value;
  const maxAttempts = 3;

  try {
    if ((imageInput.files || []).length > 0) {
      setStatus(aiStatus, t('imagePreprocessing'), 'working');
    }
    const images = await readImages();

    if (!selected) {
      setStatus(aiStatus, t('selectProviderFirst'), 'error');
      return;
    }

    if (format === 'xml') {
      const hasImageModel = Boolean(String(selected.imageModel || '').trim());
      const wantsAuto = Boolean(state.imageModelEnabled && hasImageModel);
      const wantsModify =
        images.length > 0 &&
        (/\b(style|styled|in the style|redraw|recreate|modify|variation|variant)\b/i.test(prompt) ||
          /风格|参考|改成|重绘|生成.*类似|类似.*风格|按.*风格|变体/.test(prompt));

      if (wantsAuto && (images.length === 0 || wantsModify)) {
        await generateReferenceImage();
        return;
      }

      // With reference image + XML: enter precision calibration first (task is keyed by imageHash+prompt).
      if (images.length > 0) {
        setStatus(aiStatus, t('calibPreparing'), 'working');
        await openCalibrationFromCurrentInput();
        setStatus(aiStatus, t('calibOpened'), 'success');
        return;
      }
    }

    const effectiveFormat = images.length === 0 && format === 'xml' ? 'json' : format;

    let overlays = [];
    if (images.length > 0) {
      try {
        overlays = await detectOverlays(images, selected.type);
        const imgW = Number(images[0].width || 0);
        const imgH = Number(images[0].height || 0);
        const imgArea = imgW > 0 && imgH > 0 ? imgW * imgH : 0;
        const allowedOverlayKinds = new Set(['icon', 'photo', 'chart', 'plot', '3d']);
        overlays = overlays
          .filter((ov) => ov && typeof ov === 'object')
          .filter((ov) => Number(ov.width) >= 24 && Number(ov.height) >= 24)
          .filter((ov) => (typeof ov.kind === 'string' ? allowedOverlayKinds.has(ov.kind) : true))
          .filter((ov) => (typeof ov.confidence === 'number' ? ov.confidence >= 0.5 : true))
          .filter((ov) => {
            if (!imgArea) return true;
            const area = Number(ov.width) * Number(ov.height);
            if (!Number.isFinite(area)) return true;
            const frac = area / imgArea;
            return frac >= 0.001 && frac <= 0.18;
          });
      } catch (err) {
        reportClientError(err, { phase: 'detectOverlays', providerType: selected.type });
        overlays = [];
      }
    }
    const overlayExtractedCache = new Map();
    let currentPrompt = prompt;
    let lastValidationError = null;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      setStatus(aiStatus, t('generatingDiagramAttempt', { attempt: String(attempt), max: String(maxAttempts) }), 'working');

      const payload = {
        prompt: currentPrompt,
        images,
        format: effectiveFormat,
        providerType: selected.type
      };

      const response = await fetch(`/api/flow/${effectiveFormat}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        const error = new Error(errorBody.error || 'AI request failed');
        reportClientError(error, { phase: 'callFlow', format: effectiveFormat, providerType: selected.type });
        throw error;
      }

      const data = await response.json();
      const raw = cleanAiResponse(data.output || '');
      const warningCode = data.warningCode || null;
      const warningText = data.warning || null;

      setStatus(aiStatus, t('validatingOutput'), 'working');

      if (effectiveFormat === 'json') {
        let json;
        try {
          json = JSON.parse(raw);
        } catch (err) {
          lastValidationError = `Invalid JSON: ${err.message}`;
          if (attempt < maxAttempts) {
            currentPrompt = buildRepairPrompt(prompt, effectiveFormat, raw, lastValidationError);
            continue;
          }
          throw new Error(lastValidationError);
        }

        const validation = validateReactFlowJson(json);
        if (!validation.ok) {
          lastValidationError = validation.error;
          if (attempt < maxAttempts) {
            currentPrompt = buildRepairPrompt(prompt, effectiveFormat, JSON.stringify(json), lastValidationError);
            continue;
          }
          throw new Error(lastValidationError);
        }

        let xml = sanitizeMxGraphXml(jsonToXml(json));
        if (images.length > 0 && overlays.length > 0) {
          setStatus(aiStatus, t('overlayExtracting'), 'working');
          let extracted = overlayExtractedCache.get(images[0].dataUrl);
          if (!extracted) {
            extracted = [];
            for (const overlay of overlays) {
              try {
                const out = await extractOverlayDataUrl(images[0].dataUrl, overlay);
                if (out) extracted.push(out);
              } catch (err) {
                reportClientError(err, { phase: 'extractOverlayDataUrl', providerType: selected.type });
              }
            }
            overlayExtractedCache.set(images[0].dataUrl, extracted);
          }
          xml = insertOverlaysIntoXml(xml, extracted, { imageWidth: images[0].width, imageHeight: images[0].height });
          xml = sanitizeMxGraphXml(xml);
        }
        xml = autoLayoutMxGraphXml(xml, { repositionVertices: images.length === 0 });
        xml = sanitizeMxGraphXml(xml);
        state.currentJson = json;
        try {
          await loadXmlWithGuard(xml);
        } catch (err) {
          lastValidationError = t('canvasImportError', { error: String(err.message || err) });
          if (attempt < maxAttempts) {
            currentPrompt = buildRepairPrompt(prompt, effectiveFormat, JSON.stringify(json), lastValidationError);
            continue;
          }
          throw new Error(lastValidationError);
        }
      } else {
        let xml = sanitizeMxGraphXml(normalizeXml(raw));
        if (images.length > 0 && overlays.length > 0) {
          setStatus(aiStatus, t('overlayExtracting'), 'working');
          let extracted = overlayExtractedCache.get(images[0].dataUrl);
          if (!extracted) {
            extracted = [];
            for (const overlay of overlays) {
              try {
                const out = await extractOverlayDataUrl(images[0].dataUrl, overlay);
                if (out) extracted.push(out);
              } catch (err) {
                reportClientError(err, { phase: 'extractOverlayDataUrl', providerType: selected.type });
              }
            }
            overlayExtractedCache.set(images[0].dataUrl, extracted);
          }
          xml = insertOverlaysIntoXml(xml, extracted, { imageWidth: images[0].width, imageHeight: images[0].height });
          xml = sanitizeMxGraphXml(xml);
        }
        xml = autoLayoutMxGraphXml(xml, { repositionVertices: images.length === 0 });
        xml = sanitizeMxGraphXml(xml);
        const validation = validateMxGraphXml(xml);
        if (!validation.ok) {
          lastValidationError = validation.error;
          if (attempt < maxAttempts) {
            currentPrompt = buildRepairPrompt(prompt, effectiveFormat, xml, lastValidationError);
            continue;
          }
          throw new Error(lastValidationError);
        }
        try {
          await loadXmlWithGuard(xml);
        } catch (err) {
          lastValidationError = t('canvasImportError', { error: String(err.message || err) });
          if (attempt < maxAttempts) {
            currentPrompt = buildRepairPrompt(prompt, effectiveFormat, xml, lastValidationError);
            continue;
          }
          throw new Error(lastValidationError);
        }
      }

      const warning = formatProviderWarning(warningCode, warningText);
      const suffix = warning ? ` ${warning}` : '';

      const overlaySuffix = images.length > 0 && overlays.length > 0 ? ` ${t('overlayApplied', { count: String(overlays.length) })}` : '';
      setStatus(aiStatus, `${t('diagramLoaded')}${suffix}${overlaySuffix}`, 'success');
      return;
    }

    throw new Error(t('invalidOutput', { error: lastValidationError || 'Unknown validation error' }));
  } catch (err) {
    reportClientError(err, {
      phase: 'generateDiagram',
      providerType: getSelectedProvider()?.type,
      format: document.querySelector('input[name=\"format\"]:checked')?.value || ''
    });
    setStatus(aiStatus, t('generationFailed', { error: err.message }), 'error');
  }
}

function handleMessage(event) {
  if (!event.data) return;
  let message = event.data;
  if (typeof message === 'string') {
    try {
      message = JSON.parse(message);
    } catch (err) {
      return;
    }
  }

  if (message.event === 'configure') {
    postToEditor({ action: 'configure', config: {} });
    return;
  }

  if (message.event === 'error') {
    const errorText = message.message || message.error || message.data || 'Canvas error';
    if (state.pendingCanvasLoad) {
      const pending = state.pendingCanvasLoad;
      state.pendingCanvasLoad = null;
      pending.reject(new Error(String(errorText)));
    }
    reportClientError(new Error(String(errorText)), { phase: 'canvas', event: 'error' });
    return;
  }

  if (message.event === 'autosave' && message.xml) {
    const now = Date.now();
    const xml = normalizeXml(message.xml);
    state.currentXml = xml;

    const saveNow = () => {
      lastAutosaveTs = Date.now();
      try {
        localStorage.setItem(CANVAS_XML_KEY, xml);
      } catch (err) {
        // ignore
      }
    };

    if (now - lastAutosaveTs > 1200) {
      saveNow();
    } else {
      if (autosaveTimer) {
        clearTimeout(autosaveTimer);
      }
      autosaveTimer = setTimeout(() => {
        autosaveTimer = null;
        saveNow();
      }, 1200);
    }
  }

  if (message.event === 'save' && message.xml) {
    const xml = normalizeXml(message.xml);
    state.currentXml = xml;
    try {
      localStorage.setItem(CANVAS_XML_KEY, xml);
    } catch (err) {
      // ignore
    }
  }

  if (message.event === 'load' && state.pendingCanvasLoad) {
    const pending = state.pendingCanvasLoad;
    state.pendingCanvasLoad = null;
    pending.resolve();
  }

  if (message.event === 'init') {
    // Diagrams may focus the iframe during init; counteract that and then release the guard shortly after.
    if (state.scrollGuardActive) {
      try {
        if (document.activeElement === iframe) {
          promptInput?.focus?.({ preventScroll: true });
        }
      } catch (err) {
        // ignore
      }
      window.scrollTo(0, 0);
      requestAnimationFrame(() => window.scrollTo(0, 0));
      setTimeout(() => {
        state.scrollGuardActive = false;
      }, 900);
    }
    state.iframeReady = true;
    if (state.currentXml) {
      loadXml(state.currentXml);
    } else {
      let restored = '';
      try {
        restored = localStorage.getItem(CANVAS_XML_KEY) || '';
      } catch (err) {
        restored = '';
      }

      if (restored) {
        state.currentJson = null;
        loadXml(restored);
      } else {
        const xml = jsonToXml(sampleJson);
        state.currentJson = sampleJson;
        loadXml(xml);
      }
    }
  }

  if (message.event === 'export' && state.pendingExport) {
    const pending = state.pendingExport;
    state.pendingExport = null;
    const format = pending && pending.format ? String(pending.format) : 'xml';
    if (format === 'png') {
      pending.resolve(String(message.data || '').trim());
    } else {
      const xml = normalizeXml(message.data || '');
      pending.resolve(xml);
    }
  }
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('File read failed'));
    reader.readAsText(file);
  });
}

function updateProviderFields() {
  const selected = getSelectedProvider();
  if (!selected) {
    providerFields.dataset.hidden = 'true';
    providerKey.value = '';
    providerBase.value = '';
    providerModel.value = '';
    if (providerImageModel) providerImageModel.value = '';
    return;
  }

  providerFields.dataset.hidden = 'false';
  const cachedKey = providerKeyCache[selected.type];
  providerKey.value = typeof cachedKey === 'string' ? cachedKey : '';
  providerBase.value = selected.baseUrl || selected.defaultBase || '';
  providerModel.value = selected.model || selected.defaultModel || '';
  if (providerImageModel) providerImageModel.value = selected.imageModel || '';

  const needsBase = Boolean(selected.requiresBase);
  if (providerBaseRow) {
    providerBaseRow.style.display = needsBase ? '' : 'none';
  }
}

function isProviderConfigured(selected) {
  if (!selected) return false;
  if (typeof selected.configured === 'boolean') return selected.configured;
  return false;
}

function updateProviderStatus() {
  const selected = getSelectedProvider();
  const configured = isProviderConfigured(selected);
  const name = selected ? (selected.labelKey ? t(selected.labelKey) : selected.label || selected.type) : t('providerNone');
  const statusLabel = configured ? t('providerConfigured') : t('providerNotConfigured');
  providerStatusRow.dataset.state = configured ? 'ready' : 'missing';
  providerStatusText.textContent = t('providerStatus', { name, status: statusLabel });
  providerStatusRow.querySelector('.provider-status').dataset.state = configured ? 'ready' : 'missing';
  providerSettingsShortcut.style.display = configured ? 'none' : '';
}

function openSettings() {
  settingsModal.setAttribute('aria-hidden', 'false');
}

function closeSettings() {
  settingsModal.setAttribute('aria-hidden', 'true');
}

function openVisionDebug() {
  if (!visionDebugModal) return;
  visionDebugModal.setAttribute('aria-hidden', 'false');
}

function closeVisionDebug() {
  if (!visionDebugModal) return;
  visionDebugModal.setAttribute('aria-hidden', 'true');
}

function initLanguage() {
  const stored = localStorage.getItem(LANGUAGE_KEY);
  if (stored) {
    state.language = stored;
  } else {
    state.language = navigator.language && navigator.language.startsWith('zh') ? 'zh' : 'en';
  }
  languageSelect.value = state.language;
  applyI18n();
}

function initCriticSetting() {
  let stored = '';
  try {
    stored = localStorage.getItem(CRITIC_ENABLED_KEY) || '';
  } catch (err) {
    stored = '';
  }

  if (stored === '0') state.criticEnabled = false;
  else if (stored === '1') state.criticEnabled = true;
  else state.criticEnabled = true;

  if (criticToggle) {
    criticToggle.checked = Boolean(state.criticEnabled);
    criticToggle.addEventListener('change', () => {
      state.criticEnabled = Boolean(criticToggle.checked);
      try {
        localStorage.setItem(CRITIC_ENABLED_KEY, state.criticEnabled ? '1' : '0');
      } catch (err) {
        // ignore
      }
    });
  }
}

function initOverlayTrimSetting() {
  let stored = '';
  try {
    stored = localStorage.getItem(OVERLAY_TRIM_ENABLED_KEY) || '';
  } catch (err) {
    stored = '';
  }

  if (stored === '0') state.overlayTrimEnabled = false;
  else if (stored === '1') state.overlayTrimEnabled = true;
  else state.overlayTrimEnabled = true;

  if (overlayTrimToggle) {
    overlayTrimToggle.checked = Boolean(state.overlayTrimEnabled);
    overlayTrimToggle.addEventListener('change', () => {
      state.overlayTrimEnabled = Boolean(overlayTrimToggle.checked);
      try {
        localStorage.setItem(OVERLAY_TRIM_ENABLED_KEY, state.overlayTrimEnabled ? '1' : '0');
      } catch (err) {
        // ignore
      }
    });
  }
}

function initQualityModeSetting() {
  let stored = '';
  try {
    stored = localStorage.getItem(QUALITY_MODE_KEY) || '';
  } catch (err) {
    stored = '';
  }

  if (stored === 'balanced' || stored === 'max') state.qualityMode = stored;
  else state.qualityMode = 'max';

  if (qualityModeSelect) {
    qualityModeSelect.value = state.qualityMode;
    qualityModeSelect.addEventListener('change', () => {
      const next = String(qualityModeSelect.value || '').trim();
      state.qualityMode = next === 'balanced' ? 'balanced' : 'max';
      try {
        localStorage.setItem(QUALITY_MODE_KEY, state.qualityMode);
      } catch (err) {
        // ignore
      }
    });
  }
}

function initImageModelSetting() {
  let stored = '';
  try {
    stored = localStorage.getItem(IMAGE_MODEL_ENABLED_KEY) || '';
  } catch (err) {
    stored = '';
  }

  if (stored === '1') state.imageModelEnabled = true;
  else if (stored === '0') state.imageModelEnabled = false;
  else state.imageModelEnabled = false;

  if (imageModelEnabledToggle) {
    imageModelEnabledToggle.checked = Boolean(state.imageModelEnabled);
    imageModelEnabledToggle.addEventListener('change', () => {
      state.imageModelEnabled = Boolean(imageModelEnabledToggle.checked);
      try {
        localStorage.setItem(IMAGE_MODEL_ENABLED_KEY, state.imageModelEnabled ? '1' : '0');
      } catch (err) {
        // ignore
      }
    });
  }
}

function init() {
  loadProviderKeyCache();
  initLanguage();
  initCriticSetting();
  initOverlayTrimSetting();
  initQualityModeSetting();
  initImageModelSetting();
  fetchServerProviders();

  // Prevent the diagrams.net iframe boot from stealing focus/scroll and jumping the page.
  // Keep guard active until diagrams sends 'init' (or the user interacts), with a hard cap.
  state.scrollGuardActive = true;
  state.scrollGuardUntil = Date.now() + 20000;
  let scrollGuardTicking = false;

  const disableScrollGuard = () => {
    if (!state.scrollGuardActive) return;
    state.scrollGuardActive = false;
    try {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('focusin', onFocusIn, true);
    } catch (err) {
      // ignore
    }
  };

  const applyScrollGuard = () => {
    if (!state.scrollGuardActive) return;
    if (Date.now() > (state.scrollGuardUntil || 0)) {
      disableScrollGuard();
      return;
    }
    try {
      if (document.activeElement === iframe) {
        promptInput?.focus?.({ preventScroll: true });
      }
    } catch (err) {
      // ignore
    }
    if (window.scrollY !== 0) window.scrollTo(0, 0);
  };

  const onScroll = () => {
    if (!state.scrollGuardActive) return;
    if (scrollGuardTicking) return;
    scrollGuardTicking = true;
    requestAnimationFrame(() => {
      scrollGuardTicking = false;
      applyScrollGuard();
      // Some browsers scroll after focus changes; do a second pass.
      requestAnimationFrame(() => applyScrollGuard());
    });
  };

  const onFocusIn = (e) => {
    if (!state.scrollGuardActive) return;
    if (e?.target === iframe) {
      applyScrollGuard();
    }
  };

  // Release guard as soon as the user interacts.
  window.addEventListener('wheel', disableScrollGuard, { passive: true, once: true });
  window.addEventListener('touchstart', disableScrollGuard, { passive: true, once: true });
  window.addEventListener('keydown', disableScrollGuard, { once: true });
  setTimeout(disableScrollGuard, 20500);
  window.addEventListener('scroll', onScroll, true);
  window.addEventListener('focusin', onFocusIn, true);

  try {
    if ('scrollRestoration' in history) {
      history.scrollRestoration = 'manual';
    }
  } catch (err) {
    // ignore
  }
  window.scrollTo(0, 0);
  applyScrollGuard();

  window.addEventListener('error', (event) => {
    const err = event.error || new Error(event.message || 'Window error');
    reportClientError(err, { phase: 'window.error', filename: event.filename, lineno: event.lineno, colno: event.colno });
  });

  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason || new Error('Unhandled rejection');
    const err = reason instanceof Error ? reason : new Error(String(reason));
    reportClientError(err, { phase: 'window.unhandledrejection' });
  });

  providerSelect.addEventListener('change', () => {
    state.primaryProvider = providerSelect.value;
    setSelectedProviderId(providerSelect.value);
    updateProviderStatus();
    updateProviderFields();
  });
  providerKey.addEventListener('input', () => {
    const selected = getSelectedProvider();
    if (selected) {
      providerKeyCache[selected.type] = providerKey.value;
      persistProviderKeyCache();
    }
  });

  window.addEventListener('message', handleMessage);
  iframe?.addEventListener?.('load', () => applyScrollGuard());

  generateBtn.addEventListener('click', generateDiagram);
  if (calibrateBtn) {
    calibrateBtn.addEventListener('click', () => {
      openCalibrationFromCurrentInput();
    });
  }
  if (visionDebugBtn) {
    visionDebugBtn.addEventListener('click', async () => {
      try {
        setStatus(aiStatus, t('visionDebugWorking'), 'working');
        const images = await readImages();
        if (!images || images.length === 0) return;
        const out = await fetchVisionDebug(images, promptInput.value);
        const annotated = out && typeof out.annotated === 'string' ? out.annotated : '';
        const structure = out && typeof out.structure === 'object' ? out.structure : null;
        const meta = out && typeof out.meta === 'object' ? out.meta : null;

        if (visionDebugImage) {
          visionDebugImage.src = annotated || '';
        }
        if (visionDebugMeta) {
          const n = structure && Array.isArray(structure.nodes) ? structure.nodes.length : 0;
          const e = structure && Array.isArray(structure.edges) ? structure.edges.length : 0;
          const backend = meta && meta.backend ? String(meta.backend) : '';
          visionDebugMeta.textContent = backend ? `${backend} · nodes=${n} · edges=${e}` : `nodes=${n} · edges=${e}`;
        }
        openVisionDebug();
        setStatus(aiStatus, t('idle'), 'idle');
      } catch (err) {
        setStatus(aiStatus, t('visionDebugFailed', { error: err.message }), 'error');
        reportClientError(err, { phase: 'visionDebug' });
      }
    });
  }
  clearPromptBtn.addEventListener('click', () => {
    promptInput.value = '';
    imageInput.value = '';
    renderImagePreviews();
    if (overlayFailuresPanel) overlayFailuresPanel.dataset.hidden = 'true';
    if (retryOverlaysBtn) retryOverlaysBtn.disabled = true;
    state.lastStructure = null;
    state.lastImages = null;
    state.lastPrompt = '';
    state.lastProviderType = '';
    setStatus(aiStatus, t('idle'), 'idle');
  });

  imageInput.addEventListener('change', () => {
    const files = Array.from(imageInput.files || []);
    if (files.length > 4) {
      updateImageInputFiles(files.slice(0, 4));
      renderImagePreviews(t('imageMaxKept', { max: '4' }));
      return;
    }
    renderImagePreviews();
  });

  importJsonBtn.addEventListener('click', () => handleImportJson(importText.value));
  importXmlBtn.addEventListener('click', () => handleImportXml(importText.value));

  importFile.addEventListener('change', async () => {
    const file = importFile.files[0];
    if (!file) return;
    const content = await readFileAsText(file);
    if (file.name.endsWith('.json')) {
      handleImportJson(content);
    } else {
      handleImportXml(content);
    }
  });

  exportXmlBtn.addEventListener('click', handleExportXml);
  exportJsonBtn.addEventListener('click', handleExportJson);
  syncBtn.addEventListener('click', syncFromCanvas);

  saveProviderBtn.addEventListener('click', saveProvider);
  if (getApiBtn) {
    getApiBtn.addEventListener('click', () => {
      window.open('https://0-0.pro/', '_blank', 'noopener,noreferrer');
    });
  }

  settingsBtn.addEventListener('click', openSettings);
  closeSettingsBtn.addEventListener('click', closeSettings);
  settingsModal.addEventListener('click', (event) => {
    if (event.target && event.target.hasAttribute('data-modal-close')) {
      closeSettings();
    }
  });

  if (closeVisionDebugBtn) closeVisionDebugBtn.addEventListener('click', closeVisionDebug);
  if (visionDebugModal) {
    visionDebugModal.addEventListener('click', (event) => {
      if (event.target && event.target.hasAttribute('data-modal-close')) {
        closeVisionDebug();
      }
    });
  }

  if (imageViewerCloseBtn) imageViewerCloseBtn.addEventListener('click', closeImageViewer);
  if (imageViewerModal) {
    imageViewerModal.addEventListener('click', (event) => {
      if (event.target && event.target.hasAttribute('data-modal-close')) {
        closeImageViewer();
      }
    });
  }
  if (imageViewerZoomInBtn) imageViewerZoomInBtn.addEventListener('click', () => zoomImageViewer(1.12));
  if (imageViewerZoomOutBtn) imageViewerZoomOutBtn.addEventListener('click', () => zoomImageViewer(1 / 1.12));
  if (imageViewerResetBtn) imageViewerResetBtn.addEventListener('click', resetImageViewerView);
  if (imageViewerDownloadBtn) {
    imageViewerDownloadBtn.disabled = true;
    imageViewerDownloadBtn.addEventListener('click', () => {
      if (!imageViewerState.src) return;
      downloadUrlAsFile(imageViewerState.src, imageViewerState.filename || 'image.png');
    });
  }
  if (imageViewerViewport) {
    imageViewerViewport.addEventListener('wheel', handleImageViewerWheel, { passive: false });
    imageViewerViewport.addEventListener('pointerdown', handleImageViewerPointerDown);
    imageViewerViewport.addEventListener('pointermove', handleImageViewerPointerMove);
    imageViewerViewport.addEventListener('pointerup', handleImageViewerPointerUp);
    imageViewerViewport.addEventListener('pointercancel', handleImageViewerPointerUp);
    imageViewerViewport.addEventListener('dblclick', resetImageViewerView);
  }

  if (calibCloseBtn) calibCloseBtn.addEventListener('click', closeCalibration);
  if (calibrationModal) {
    calibrationModal.addEventListener('click', (event) => {
      if (event.target && event.target.hasAttribute('data-modal-close')) {
        closeCalibration();
      }
    });
  }

  if (calibToggleHistoryBtn && calibHistoryPanel) {
    calibToggleHistoryBtn.addEventListener('click', () => {
      const collapsed = calibHistoryPanel.dataset.collapsed === 'true';
      calibHistoryPanel.dataset.collapsed = collapsed ? 'false' : 'true';
    });
  }

  if (calibViewImageBtn) {
    calibViewImageBtn.addEventListener('click', () => {
      const img = calibState.currentTask?.image;
      const href = img && typeof img.dataUrl === 'string' ? img.dataUrl : '';
      if (!href) return;
      openImageViewer(href, img?.name || t('referenceImage'));
    });
  }
  if (calibDownloadImageBtn) {
    calibDownloadImageBtn.addEventListener('click', () => {
      const img = calibState.currentTask?.image;
      const href = img && typeof img.dataUrl === 'string' ? img.dataUrl : '';
      if (!href) return;
      downloadUrlAsFile(href, img?.name || 'reference.png');
    });
  }

  if (calibToolSelectBtn) calibToolSelectBtn.addEventListener('click', () => setCalibTool('select'));
  if (calibToolNodeBtn) calibToolNodeBtn.addEventListener('click', () => setCalibTool('newNode'));
  if (calibToolNewBtn) calibToolNewBtn.addEventListener('click', () => setCalibTool('newOverlay'));
  if (calibToolFgBtn) calibToolFgBtn.addEventListener('click', () => setCalibTool('fg'));
  if (calibToolBgBtn) calibToolBgBtn.addEventListener('click', () => setCalibTool('bg'));
  if (calibZoomResetBtn) calibZoomResetBtn.addEventListener('click', resetCalibView);
  if (calibUndoBtn) calibUndoBtn.addEventListener('click', performCalibUndo);
  if (calibRedoBtn) calibRedoBtn.addEventListener('click', performCalibRedo);

  if (calibViewport) {
    calibViewport.addEventListener('wheel', handleCalibWheel, { passive: false });
    calibViewport.addEventListener('pointerdown', handleCalibPointerDown);
    calibViewport.addEventListener('pointermove', handleCalibPointerMove);
    calibViewport.addEventListener('pointerup', handleCalibPointerUp);
    calibViewport.addEventListener('pointercancel', handleCalibPointerUp);
  }

  window.addEventListener('keydown', (event) => handleCalibKey(event, true));
  window.addEventListener('keyup', (event) => handleCalibKey(event, false));

  if (calibNodeShape) calibNodeShape.addEventListener('change', updateSelectedNodeFromInspector);
  if (calibNodeText) calibNodeText.addEventListener('change', updateSelectedNodeFromInspector);
  [calibNodeX, calibNodeY, calibNodeW, calibNodeH].forEach((el) => {
    if (!el) return;
    el.addEventListener('change', updateSelectedNodeFromInspector);
  });
  if (calibNodeDeleteBtn) calibNodeDeleteBtn.addEventListener('click', deleteSelectedNode);

  if (calibOvKind) calibOvKind.addEventListener('change', updateSelectedOverlayFromInspector);
  if (calibOvOwner) calibOvOwner.addEventListener('change', updateSelectedOverlayFromInspector);
  document.querySelectorAll('input[name=\"calib-granularity\"]').forEach((el) => {
    el.addEventListener('change', updateSelectedOverlayFromInspector);
  });
	  [calibOvX, calibOvY, calibOvW, calibOvH].forEach((el) => {
	    if (!el) return;
	    el.addEventListener('change', updateSelectedOverlayFromInspector);
	  });
	
	  if (calibOvClearPointsBtn) calibOvClearPointsBtn.addEventListener('click', clearSelectedOverlayPoints);
	  if (calibOvSegmentBtn) calibOvSegmentBtn.addEventListener('click', segmentSelectedOverlay);
	  if (calibOvSelectOwnerBtn) calibOvSelectOwnerBtn.addEventListener('click', selectOwnerNodeFromOverlaySelection);
	  if (calibOvDeleteBtn) calibOvDeleteBtn.addEventListener('click', deleteSelectedOverlay);
	  if (calibApplyBtn) calibApplyBtn.addEventListener('click', applyCalibrationToCanvas);

  languageSelect.addEventListener('change', () => {
    state.language = languageSelect.value;
    localStorage.setItem(LANGUAGE_KEY, state.language);
    applyI18n();
  });

  providerSettingsShortcut.addEventListener('click', openSettings);
  if (toggleKeyVisibilityBtn) {
    toggleKeyVisibilityBtn.addEventListener('click', toggleKeyVisibility);
  }

  if (retryOverlaysBtn) retryOverlaysBtn.disabled = true;

  if (retryOverlaysBtn) {
    retryOverlaysBtn.addEventListener('click', async () => {
      try {
        const selected = getSelectedProvider();
        const images = state.lastImages;
        const structure = state.lastStructure;
        const prompt = state.lastPrompt || promptInput.value.trim();
        const providerType = state.lastProviderType || (selected ? selected.type : '');
        if (!images || !images[0] || !structure) {
          throw new Error('No previous structure to retry.');
        }
        setStatus(aiStatus, t('retryingOverlays'), 'working');
        retryOverlaysBtn.disabled = true;

        const response = await fetch('/api/vision/overlays/retry', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...(providerType ? { providerType } : {}),
            prompt: String(prompt || ''),
            images: [images[0]],
            imageWidth: images[0].width || 0,
            imageHeight: images[0].height || 0,
            qualityMode: state.qualityMode,
            overlayOptions: { tightenBbox: Boolean(state.overlayTrimEnabled) },
            structure
          })
        });

        if (!response.ok) {
          const msg = await readErrorMessage(response);
          throw new Error(msg || 'Retry failed overlays failed.');
        }

        const data = await response.json();
        const nextStructure = data && typeof data.structure === 'object' ? data.structure : null;
        const meta = data && typeof data.meta === 'object' ? data.meta : null;
        if (!nextStructure) throw new Error('Invalid retry response');

        const rendered = await structureToXml(nextStructure, images[0]);
        let xml = typeof rendered === 'string' ? rendered : rendered.xml;
        xml = sanitizeMxGraphXml(xml);
        xml = autoLayoutMxGraphXml(xml, { repositionVertices: false });
        xml = sanitizeMxGraphXml(xml);
        const validation = validateMxGraphXml(xml);
        if (!validation.ok) throw new Error(validation.error);
        await loadXmlWithGuard(xml);

        state.lastStructure = nextStructure;
        renderOverlayFailures(meta);
        const backendTag = meta && meta.backend ? ` (${String(meta.backend)})` : '';
        setStatus(aiStatus, `${t('diagramLoaded')}${backendTag}`, 'success');
      } catch (err) {
        setStatus(aiStatus, t('retryOverlaysFailed', { error: err.message }), 'error');
        reportClientError(err, { phase: 'retryFailedOverlays' });
      } finally {
        if (retryOverlaysBtn) retryOverlaysBtn.disabled = false;
      }
    });
  }

  updateProviderFields();
}

init();
