export const RUNTIME_BASE = new URL('./runtime/', import.meta.url).href;
let worker = null, ready = false, preparation = null, flight = null;
const listeners = new Set();
export function onLocalProgress(fn) { listeners.add(fn); return ()=>listeners.delete(fn); }
function notify(record) { for (const fn of listeners) fn(record); }
function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('browser-worker.js', import.meta.url));
  worker.onmessage = ({data}) => {
    if (data.type === 'progress') notify(data);
    if (data.type === 'ready') { ready=true; preparation?.resolve(data); preparation=null; notify({status:'ready',message:'本地引擎已就绪 · 完整 baseline / A4'}); }
    if (data.type === 'result') { flight?.resolve(data.result); flight=null; }
    if (data.type === 'error') {
      const error = Error(data.message); console.error('Local flight:',data.detail);
      preparation?.reject(error); preparation=null; flight?.reject(error); flight=null;
      worker.terminate(); worker=null; ready=false;
    }
  };
  worker.onerror = event => {
    const error=Error(event.message || '浏览器计算进程中断。草图仍在本机，可重新开始。');
    preparation?.reject(error); preparation=null; flight?.reject(error); flight=null;
    worker.terminate(); worker=null; ready=false;
  };
  return worker;
}
export async function prepareLocalFlight() {
  await offlineReady;
  if (ready) return;
  if (preparation) return preparation.promise;
  let resolve,reject; const promise=new Promise((a,b)=>{resolve=a;reject=b;});
  preparation={promise,resolve,reject};
  ensureWorker().postMessage({type:'prepare',runtimeBase:RUNTIME_BASE});
  return promise;
}
export async function runLocalFlight(request) {
  if (flight) throw Error('已有试飞正在计算');
  await prepareLocalFlight();
  const id = crypto.randomUUID();
  return new Promise((resolve,reject)=>{flight={resolve,reject};ensureWorker().postMessage({type:'flight',runtimeBase:RUNTIME_BASE,request:{...request,id}});});
}
export function cancelLocalFlight() { worker?.postMessage({type:'cancel'}); }
export function localFlightReady() { return ready; }
export async function saveLocalResult(result) {
  try {
    result.created = Date.now();
    const db = await new Promise((resolve,reject)=>{
      const req=indexedDB.open('gift-local-flights',1);
      req.onupgradeneeded=()=>req.result.createObjectStore('flights',{keyPath:'id'});
      req.onsuccess=()=>resolve(req.result); req.onerror=()=>reject(req.error);
    });
    await new Promise((resolve,reject)=>{const tx=db.transaction('flights','readwrite');tx.objectStore('flights').put(result);tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);});
    db.close(); return true;
  } catch { return false; }
}
export async function localRecords(id) {
  const db=await new Promise((resolve,reject)=>{const req=indexedDB.open('gift-local-flights',1);req.onupgradeneeded=()=>req.result.createObjectStore('flights',{keyPath:'id'});req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);});
  try {
    return await new Promise((resolve,reject)=>{
      const store=db.transaction('flights','readonly').objectStore('flights');
      if(id){const req=store.get(id);req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);return;}
      const rows=[],req=store.openCursor();
      req.onerror=()=>reject(req.error);req.onsuccess=()=>{const cursor=req.result;if(!cursor){resolve(rows.sort((a,b)=>b.created-a.created));return;}
        rows.push({id:cursor.value.id,created:cursor.value.created||0,meta:cursor.value.meta});cursor.continue();};
    });
  } finally { db.close(); }
}
const offlineReady = 'serviceWorker' in navigator ? (async()=>{
  try {
    await navigator.serviceWorker.register(new URL('browser-sw.js', import.meta.url), {scope:'./'});
    await navigator.serviceWorker.ready;
    if (!navigator.serviceWorker.controller) await new Promise(resolve=>navigator.serviceWorker.addEventListener('controllerchange',resolve,{once:true}));
  } catch(error) { console.warn('离线页面缓存未能启用',error); }
})() : Promise.resolve();
