import {SpatialEditor} from './editor3d.js?v=14';
import {sampleCurve} from './smooth_curve.js?v=6';
import {runLocalFlight,cancelLocalFlight,onLocalProgress,prepareLocalFlight,saveLocalResult,localRecords,localFlightReady} from './browser-flight.js';
const $=id=>document.getElementById(id),copy=v=>JSON.parse(JSON.stringify(v));
let points=[],selected=2,undo=[],redo=[],transaction=null,reference=[],controlSegments=[],job=null,version=0,flightVersion=-1,compilation=null,previewIssues=[],risk=[];
const templates={
 s:[[0,0,5000],[1000,0,5000],[4000,600,5200],[7000,-600,5350],[10000,300,5150],[13000,300,5150]],
 arc:[[0,0,5000],[1000,0,5000],[4000,700,5200],[6500,2400,5450],[7800,5000,5650],[8000,7800,5650]],
 uturn:[[0,0,5000],[1000,0,5000],[4000,900,5200],[6000,3000,5450],[4500,5600,5400],[1500,6500,5200],[-1500,6500,5200]]
};
function begin(){if(!transaction)transaction=copy(points);}
function commit(){if(transaction){undo.push(transaction);undo=undo.slice(-40);transaction=null;redo=[];}changed();}
function message(m){$('job-status').textContent=m;}
const editor=new SpatialEditor($('editor'),{select(i){selected=i;fields();},begin,commit,abort(){transaction=null;},
 move(i,p){points[i]=p;changed(false);},add(p){begin();points.push(p);selected=points.length-1;commit();},message});
window.__editor=editor;
function previewDiagnostics(){
 const arc=[0],deltas=[],lengths=[];
 for(let i=1;i<reference.length;i++){const d=reference[i].map((v,k)=>v-reference[i-1][k]),length=Math.hypot(...d);deltas.push(d);lengths.push(length);arc.push(arc.at(-1)+length);}
 const masks={curve:[],slope:[]};
 for(let i=0;i<deltas.length;i++){
  const d=deltas[i],slope=Math.abs(Math.atan2(d[2],Math.hypot(d[0],d[1]))*180/Math.PI);
  masks.slope[i]=slope>35;
  if(i){const b=deltas[i-1],dot=d.reduce((sum,v,k)=>sum+v*b[k],0)/(lengths[i]*lengths[i-1]);const angle=Math.acos(Math.max(-1,Math.min(1,dot)));masks.curve[i]=angle*170*170/(9.80665*(lengths[i]+lengths[i-1])/2)>7.5;}
 }
 const issues=[];
 for(const [code,mask]of Object.entries(masks)){
  for(let i=0;i<mask.length;i++){if(!mask[i])continue;const start=Math.max(0,i-1);while(i+1<mask.length&&mask[i+1])i++;const end=Math.min(reference.length-1,i+1),mid=Math.floor((start+end)/2);
   const controls=Array.from(new Set(controlSegments.slice(start,end+1).flatMap(i=>[i,i+1]))).sort((a,b)=>a-b);
   issues.push({code,title:code==='curve'?'急弯或高度突变':'陡坡或垂直段',message:code==='curve'?'可拉开附近控制点以减小偏差；当前路线可以提交试飞。':'可拉长此段或减小高度差；当前路线可以提交试飞。',sample_start:start,sample_end:end,station_start_m:arc[start],station_end_m:arc[end],position:reference[mid],point_indices:controls});
  }
 }
 return {length:arc.at(-1),issues};
}
function showDiagnostics(){
 const issues=compilation?.issues||previewIssues;
 $('diagnostic-summary').textContent=compilation?.message||(issues.length?`草图提示：${issues.length} 处急弯或陡坡，可直接试飞。`:'草图可直接试飞；编译后在这里查看路段提示。');
 $('diagnostic-list').replaceChildren(...issues.map((issue,index)=>{
  const button=document.createElement('button');button.className='diagnostic-item';button.type='button';
  const title=document.createElement('strong');title.textContent=`${index+1}. ${issue.title}`;
  const location=document.createElement('span');const indices=issue.point_indices||[];
  location.textContent=`沿路线 ${(issue.station_start_m/1000).toFixed(2)}—${(issue.station_end_m/1000).toFixed(2)} km${indices.length?' · 控制点 '+indices.join('、'):''}`;
  const detail=document.createElement('small');detail.textContent=issue.message;
  button.append(title,location,detail);button.onclick=()=>{
   $('diagnostic-list').querySelectorAll('button').forEach(b=>b.classList.remove('selected'));button.classList.add('selected');
   if(indices.length){selected=indices[Math.floor(indices.length/2)];selected=Math.min(selected,points.length-1);editor.select(selected);fields();}
   editor.focusIssue(issue.position,reference.slice(issue.sample_start,issue.sample_end+1));$('editor').scrollIntoView({behavior:'smooth',block:'center'});
   message(`${issue.title} · 北 ${Math.round(issue.position[0])}、东 ${Math.round(issue.position[1])}、高 ${Math.round(issue.position[2])} m。${issue.message}`);
  };return button;
 }));
 risk=reference.map((p,i)=>issues.some(v=>i>=v.sample_start&&i<=v.sample_end)?p:null);
 $('diagnostic-legend').hidden=!issues.length;
 editor.update(points,selected,reference,risk,[],!job);
 window.__compilation=compilation;
}
function buildReference(){
 try{const sampled=sampleCurve(points);reference=sampled.positions;controlSegments=sampled.controlSegments;}
 catch(e){reference=points;previewIssues=[];risk=[];compilation=null;$('diagnostic-list').replaceChildren();$('diagnostic-summary').textContent=e.message;$('geometry-warning').textContent=e.message;$('run').disabled=true;editor.update(points,selected,reference,[],[],!job);return;}
 const d=previewDiagnostics();previewIssues=d.issues;let low=Infinity,high=-Infinity;
 for(const p of reference){low=Math.min(low,p[2]);high=Math.max(high,p[2]);}
 $('geometry-summary').textContent=`长度 ${(d.length/1000).toFixed(2)} km · 高度 ${Math.round(low)}—${Math.round(high)} m · ${points.length} 个控制点`;
 $('geometry-warning').textContent='';$('run').disabled=!!job;showDiagnostics();
}
function fields(){
 const p=points[selected];if(!p)return;$('point-title').textContent=`控制点 ${selected}${selected===0?' · 飞行起点':''}`;
 ['n','e','h'].forEach((k,i)=>{$('point-'+k).value=Math.round(p[i]);$('point-'+k).disabled=!!job;});
 $('remove').disabled=points.length<=2||!!job;$('insert').disabled=!!job;
 $('point-list').replaceChildren(...points.map((_,i)=>{const b=document.createElement('button');b.textContent=i;b.className=i===selected?'selected':'';b.onclick=()=>{selected=i;editor.select(i);fields();};return b;}));
}
function changed(dirty=true){
 compilation=null;editor.clearIssueFocus();
 if(dirty){version++;try{localStorage.setItem('gift-browser-points-v1',JSON.stringify(points));}catch{message('草图可继续编辑与试飞；浏览器存储已满，可导出路线保存。');}}
 buildReference();fields();if(flightVersion>=0&&version!==flightVersion)$('custom-result').hidden=true;
}
function reset(){if(job)return;begin();points=copy(templates[$('template').value]);selected=2;commit();editor.setBounds(reference);}
$('template').onchange=reset;$('reset-template').onclick=reset;
$('undo').onclick=()=>{if(!undo.length||job)return;redo.push(copy(points));points=undo.pop();selected=Math.min(selected,points.length-1);changed();};
$('redo').onclick=()=>{if(!redo.length||job)return;undo.push(copy(points));points=redo.pop();selected=Math.min(selected,points.length-1);changed();};
['n','e','h'].forEach((k,i)=>$('point-'+k).onchange=e=>{if(job)return;const v=Number(e.target.value);if(!Number.isFinite(v)||v===points[selected][i])return;begin();points[selected][i]=v;commit();});
$('insert').onclick=()=>{if(job)return;begin();const i=selected,a=points[i],b=points[i+1]||a.map((v,k)=>v+(v-points[Math.max(0,i-1)][k]));points.splice(i+1,0,a.map((v,k)=>(v+b[k])/2));selected=i+1;commit();};
$('remove').onclick=()=>{if(job||points.length<=2)return;begin();points.splice(selected,1);selected=Math.min(selected,points.length-1);commit();};
$('fit').onclick=()=>editor.setBounds(reference);
$('add-mode').onchange=e=>editor.setMode(e.target.checked?'add':'select');
$('draw-plane').onchange=()=>{const v=$('draw-plane').value;$('plane-value').value=v==='NE'?5000:0;$('plane-label').firstChild.textContent=v==='NE'?'平面高度 / m':v==='NH'?'固定东向 / m':'固定北向 / m';editor.setPlane(v,Number($('plane-value').value));};
$('plane-value').onchange=()=>editor.setPlane($('draw-plane').value,Number($('plane-value').value));
document.querySelectorAll('[data-editor-view]').forEach(b=>b.onclick=()=>editor.view(b.dataset.editorView));
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||job)return;if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();$('remove').click();}if((e.ctrlKey||e.metaKey)&&e.key==='z'){e.preventDefault();$('undo').click();}});
function download(name,obj){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),5000);}
$('export-route').onclick=()=>download('自定义三维路线.json',{schema:'a4.custom.v4',points});
$('import-route').onchange=async e=>{try{if(job||!e.target.files[0])return;const v=JSON.parse(await e.target.files[0].text());if(!Array.isArray(v.points)||v.points.length<2||v.points.some(p=>!Array.isArray(p)||p.length!==3||p.some(x=>typeof x!=='number'||!Number.isFinite(x))))throw Error('需要至少两个有限三维控制点');begin();points=v.points;selected=Math.min(2,points.length-1);commit();editor.setBounds(reference);message('路线已导入');}catch(e){message('导入失败：'+e.message);}finally{e.target.value='';}};
function showProgress(record){
 const stages=['downloading','compiling','loading','flying','completed'],index=stages.indexOf(record.status);
 $('flight-progress').hidden=false;
 for(const item of $('flight-progress').querySelectorAll('[data-stage]')){
  const n=stages.indexOf(item.dataset.stage);item.classList.toggle('current',n===index);item.classList.toggle('done',index>n);
 }
 let detail=record.message||'';
 if(record.total)detail+=` · ${(record.downloaded/1048576).toFixed(1)} / ${(record.total/1048576).toFixed(1)} MB`;
 if(record.controls!==undefined)detail+=` · ${record.controls} 步 / ${Number(record.seconds||0).toFixed(1)} 秒仿真`;
 $('stage-detail').textContent=detail;
 if(record.compilation){compilation=record.compilation;showDiagnostics();}
}
onLocalProgress(record=>{showProgress(record);if(record.status==='ready')$('service-state').textContent='本地引擎已就绪 · 可离线试飞';});
function lock(active){for(const id of ['template','reset-template','undo','redo','add-mode','insert','remove','import-route'])$(id).disabled=active;editor.editable=!active;editor.select(selected);$('cancel').hidden=!active;$('cancel').disabled=job==='preparing';$('run').textContent=active?'本机正在计算…':'开始试飞';fields();buildReference();}
$('run').onclick=async()=>{
 if(job)return;job='preparing';lock(true);message('准备本地引擎。首次下载后缓存；路线保留在本机。');
 try{
  await prepareLocalFlight();job='flying';lock(true);flightVersion=version;
  const result=await runLocalFlight({name:'自定义三维路线',positions:reference,control_points:points,control_segments:controlSegments});
  if(result.compilation){compilation=result.compilation;showDiagnostics();}
  job=null;lock(false);showProgress({status:'completed',message:'本机飞行计算完成'});
  if(result.trace?.position?.length){
   $('custom-result').hidden=false;await window.showCustomFlight(result);$('custom-result').scrollIntoView({behavior:'smooth'});
   const saved=await saveLocalResult(result);if(saved)await refreshLocalRecords();message((result.meta.note||'本机实际飞行记录已生成')+(saved?' · 已保存在此浏览器':' · 请下载记录保存'));
  }else message(result.message||'没有生成实际飞行记录');
 }catch(error){job=null;lock(false);message(error.message);}
};
$('cancel').onclick=()=>{cancelLocalFlight();message('正在结束本机计算，保留已经实际计算的航迹。');};
$('connect-service').hidden=true;
async function refreshLocalRecords(){
 try{const rows=await localRecords();$('local-records').replaceChildren(new Option('选择本机飞行记录',''),...rows.map(r=>new Option(new Date(r.created).toLocaleString()+' · '+r.meta.name+' · '+Number(r.meta.seconds).toFixed(1)+' s',r.id)));$('local-history').hidden=!rows.length;}catch{}
}
$('local-records').onchange=async e=>{if(!e.target.value||job)return;const result=await localRecords(e.target.value);if(result){flightVersion=version;$('custom-result').hidden=false;await window.showCustomFlight(result);message('正在查看本机保存的实际飞行。');}};
$('download-local').onclick=async()=>{const id=$('local-records').value;if(!id){message('请先选择一条本机记录。');return;}const result=await localRecords(id);if(result)download('GIFT-本机飞行-'+id+'.json',result);};
refreshLocalRecords();
points=copy(templates.s);try{const v=JSON.parse(localStorage.getItem('gift-browser-points-v1')||localStorage.getItem('a4-custom-points-v3')||localStorage.getItem('gift-tensorrt-points-v1')||localStorage.getItem('gift-onnx-points-v1')||localStorage.getItem('gift-custom-points-v1'));if(Array.isArray(v)&&v.length>=2&&v.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite)))points=v;}catch{}
selected=Math.min(selected,points.length-1);changed(false);editor.setBounds(reference);
$('service-state').textContent='浏览器本地计算 · 无需连接实验服务器';
window.__customReady=true;
