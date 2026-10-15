/* All flight computation stays in this Web Worker. No job API or server transport. */
const CACHE = 'gift-browser-assets-20261015-v1';
let base, manifest, python, initialized, running = false;
const sessions = new Map();
const emit = (type, values = {}) => postMessage({type, ...values});
self.gift_cancelled = false;
self.gift_progress = data => emit('progress', JSON.parse(data));

async function bytes(path, expectedHash) {
  const url = new URL(path, base).href;
  const cache = await caches.open(CACHE);
  let response = await cache.match(url);
  if (!response) {
    response = await fetch(url, {credentials:'same-origin'});
    if (!response.ok) throw Error(`无法下载本地运行资源：${path} (${response.status})`);
    await cache.put(url, response.clone());
  }
  const buffer = await response.arrayBuffer();
  if (expectedHash) {
    const actual = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), n=>n.toString(16).padStart(2,'0')).join('');
    if (actual !== expectedHash) { await cache.delete(url); throw Error(`资源校验未通过：${path}，请重新准备本地引擎。`); }
  }
  return buffer;
}

async function getSession(key) {
  if (sessions.has(key)) return sessions.get(key);
  const graph = manifest.graphs.find(g=>g.key === key || g.name === key || g.name === key+'.onnx');
  if (!graph) throw Error('模型图未列入冻结清单：'+key);
  const data = new Uint8Array(graph.bytes); let offset = 0;
  for (const part of graph.parts) { const chunk = new Uint8Array(await bytes(part.path, part.sha256)); data.set(chunk, offset); offset += chunk.length; }
  const session = await ort.InferenceSession.create(data, {
    executionProviders:['wasm'], graphOptimizationLevel:'all',
    executionMode:'sequential', enableCpuMemArena:true, enableMemPattern:true,
  });
  sessions.set(key, session);
  return session;
}

self.gift_ort_run = async (key, inputJson) => {
  const session = await getSession(key), inputs = JSON.parse(inputJson), feeds = {};
  for (const [index, item] of inputs.entries()) {
    const name = item.name || session.inputNames[index], type = item.type || 'float32';
    const dtype = {float32:Float32Array, float64:Float64Array, int32:Int32Array, int64:BigInt64Array, bool:Uint8Array}[type];
    if (!dtype) throw Error('不支持的冻结图输入类型：'+type);
    const values = type === 'int64' ? item.data.map(BigInt) : item.data;
    feeds[name] = new ort.Tensor(type, new dtype(values), item.dims);
  }
  const outputs = await session.run(feeds);
  return JSON.stringify(session.outputNames.map(name=>({name, type:outputs[name].type,
    dims:outputs[name].dims, data:Array.from(outputs[name].data, v=>typeof v==='bigint'?Number(v):v)})));
};

async function initialize(runtimeBase) {
  if (typeof WebAssembly.Suspending !== 'function' || typeof WebAssembly.promising !== 'function')
    throw Error('本地完整策略需要支持 WebAssembly JSPI 的浏览器。请使用新版 Chrome 或 Edge。');
  base = new URL(runtimeBase, self.location.href);
  const response = await fetch(new URL('manifest.json', base), {credentials:'same-origin'});
  if (!response.ok) throw Error('本地引擎清单未能加载');
  manifest = await response.json();
  const assets = manifest.assets, total = assets.reduce((n,f)=>n+f.bytes,0); let done = 0;
  emit('progress',{status:'downloading',message:'准备浏览器运行资源',downloaded:0,total});
  // Keep downloaded weights in CacheStorage, not a second in-memory model copy.
  for (const asset of assets) {
    await bytes(asset.path,asset.sha256); done += asset.bytes;
    emit('progress',{status:'downloading',message:'缓存完整 baseline 与仿真环境',downloaded:done,total});
  }
  emit('progress',{status:'loading',message:'在本机加载 Python / NumPy 与 FP32 推理引擎'});
  importScripts(new URL('ort/ort.wasm.min.js',base).href, new URL('pyodide/pyodide.js',base).href);
  ort.env.wasm.wasmPaths = new URL('ort/',base).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  python = await loadPyodide({indexURL:new URL('pyodide/',base).href});
  await python.loadPackage('numpy');
  for (const file of manifest.python_files) {
    const path = '/gift/'+file.target;
    python.FS.mkdirTree(path.slice(0,path.lastIndexOf('/')));
    python.FS.writeFile(path,new Uint8Array(await bytes(file.path,file.sha256)));
  }
  await python.runPythonAsync('import sys\nsys.path.insert(0, "/gift")\nimport browser_flight\n');
  emit('ready',{checkpoint:manifest.checkpoint,bytes:total,graphs:manifest.graphs.length});
}

self.onmessage = async event => {
  const m = event.data;
  if (m.type === 'cancel') { self.gift_cancelled = true; return; }
  try {
    if (!initialized) initialized = initialize(m.runtimeBase);
    await initialized;
    if (m.type === 'prepare') return;
    if (m.type !== 'flight' || running) return;
    running = true; self.gift_cancelled = false;
    python.globals.set('gift_request_json',JSON.stringify(m.request));
    const result = await python.runPythonAsync('await browser_flight.run_json(gift_request_json)');
    emit('result',{result:JSON.parse(result)});
  } catch (error) {
    emit('error',{message:error.message || String(error),detail:String(error.stack || '')});
  } finally { running = false; }
};
