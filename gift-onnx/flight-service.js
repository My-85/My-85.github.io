import {DEPLOYMENT_MODE,SERVICE_ORIGIN,PAGES_ORIGIN} from './site-config.js?v=14';

export const isPublicPages = DEPLOYMENT_MODE === 'pages';
const channel = 'gift-onnx-flight-v1';
const pending = new Map(), listeners = new Set();
let popup = null, nonce = '', ready = false, connecting = null, sequence = 0;

async function sameOriginAPI(path,data){
 const r=await fetch('api/lab/'+path,{method:data===undefined?'GET':'POST',headers:data===undefined?{}:{'Content-Type':'application/json'},body:data===undefined?undefined:JSON.stringify(data),credentials:'same-origin'});
 let value;try{value=await r.json();}catch{throw Error('试飞服务暂未连接，请稍后重试。');}
 if(!r.ok||value.error)throw Error(value.error||'试飞服务请求失败');
 return value;
}

function updateConnection(value){ready=value;for(const fn of listeners)fn(value);}
export function onServiceConnection(fn){listeners.add(fn);return ()=>listeners.delete(fn);}
export function serviceConnected(){return !isPublicPages||ready&&popup&&!popup.closed;}

function disconnect(message){
 updateConnection(false);
 if(connecting){clearTimeout(connecting.timer);connecting.reject(Error(message));connecting=null;}
 for(const request of pending.values())request.reject(Error(message));
 pending.clear();
}

if(isPublicPages){
 window.addEventListener('message',event=>{
  if(event.origin!==SERVICE_ORIGIN||event.source!==popup)return;
  const m=event.data;
  if(!m||m.channel!==channel||m.nonce!==nonce)return;
  if(m.type==='ready'){
   updateConnection(true);
   if(connecting){clearTimeout(connecting.timer);connecting.resolve();connecting=null;}
  }else if(m.type==='response'){
   const request=pending.get(m.id);if(!request)return;
   pending.delete(m.id);m.error?request.reject(Error(m.error)):request.resolve(m.value);
  }
 });
 setInterval(()=>{if(popup&&popup.closed&&(ready||connecting||pending.size))disconnect('试飞连接已关闭；任务编号已保留，请重新连接试飞服务。');},1000);
}

export function connectFlightService(){
 if(!isPublicPages)return Promise.resolve();
 if(serviceConnected())return Promise.resolve();
 if(connecting){popup?.focus();return connecting.promise;}
 nonce=crypto.randomUUID().replaceAll('-','');
 const url=new URL('/connect.html',SERVICE_ORIGIN);
 url.searchParams.set('origin',PAGES_ORIGIN);url.searchParams.set('nonce',nonce);
 popup=window.open(url.href,'gift-onnx-flight-service','popup,width=520,height=540');
 if(!popup)return Promise.reject(Error('请允许打开试飞连接窗口，然后重试。'));
 let resolve,reject;
 const promise=new Promise((a,b)=>{resolve=a;reject=b;});
 const timer=setTimeout(()=>{if(connecting)disconnect('尚未建立试飞连接。请在连接窗口完成登录，然后重新连接。');},90000);
 connecting={promise,resolve,reject,timer};
 return promise;
}

export async function flightAPI(path,data){
 if(!isPublicPages)return sameOriginAPI(path,data);
 if(path==='state'&&!serviceConnected())return {online:null,requires_connection:true};
 if(!serviceConnected()){
  if(connecting)await connecting.promise;
  else throw Error('请先连接试飞服务；草图和任务编号已保留。');
 }
 const id=++sequence;
 return new Promise((resolve,reject)=>{
  pending.set(id,{resolve,reject});
  try{popup.postMessage({channel,type:'request',nonce,id,path,data},SERVICE_ORIGIN);}
  catch{pending.delete(id);reject(Error('试飞连接已断开，请重新连接。'));}
 });
}

export function startServiceBridge(){
 const parameters=new URLSearchParams(location.search);
 const origin=parameters.get('origin'),session=parameters.get('nonce');
 const status=document.getElementById('bridge-status');
 if(isPublicPages||origin!==PAGES_ORIGIN||!session||!/^[a-f0-9]{32}$/.test(session)||!window.opener){
  status.textContent='请从 GIFT 轨迹编辑页面连接试飞服务。';return;
 }
 const parent=window.opener;
 const send=value=>parent.postMessage({channel,nonce:session,...value},origin);
 window.addEventListener('message',async event=>{
  const m=event.data;
  if(event.origin!==origin||event.source!==parent||!m||m.channel!==channel||m.type!=='request'||m.nonce!==session)return;
  const get=m.data===undefined;
  const valid=get?(m.path==='state'||/^jobs\/[a-f0-9-]{36}$/.test(m.path)):(m.path==='jobs'||m.path==='cancel');
  if(!valid){send({type:'response',id:m.id,error:'不支持该试飞请求。'});return;}
  try{
   const value=await sameOriginAPI(m.path,m.data);
   if(m.path==='state')status.textContent=value.online?'试飞服务在线 · ONNX FP32 已就绪':'试飞通道正在重连。可先提交路线，恢复后开始计算。';
   else status.textContent=value.message||'连接正常，试飞进度将在原页面更新。';
   send({type:'response',id:m.id,value});
  }catch(error){status.textContent=error.message;send({type:'response',id:m.id,error:error.message});}
 });
 send({type:'ready'});
 status.textContent='连接已建立。保持此窗口打开，回到 GIFT 页面即可试飞。';
 document.getElementById('return-to-editor').onclick=()=>parent.focus();
}
