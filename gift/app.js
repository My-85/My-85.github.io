import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {aircraftMesh,acmiDisplayModels} from './aircraft.js';

const $=id=>document.getElementById(id);
const statusText={pass:'通过',fail:'未通过',refused:'未飞行',recorded:'实飞记录'};
const sets={new:'候选 6ec · 训练路线 · 未晋级',keeper:'保留模型 9a · 全部评测记录',coverage:'候选 6ec · 成组诊断',aerobatics:'精选实飞 · 每类保留一条跟踪较好的代表记录。',custom:'保留模型 9a · 本次自定义试飞'};
const replayDataset=row=>row.historical?'aerobatics':row.set;
let catalog,selected,trace,playing=false,clock=0,last=performance.now(),request=0,frame=0;
let scene,camera,renderer,controls,group,plane,actualLine,span=10,center=new THREE.Vector3();
const fmt=(x,n=1)=>Number.isFinite(x)?x.toFixed(n):'—';
function text(id,value){$(id).textContent=value;}
function world(p){return new THREE.Vector3(p[1]/1000,p[2]/1000,-p[0]/1000);}
function dispose(g){g.traverse(o=>{o.geometry?.dispose();if(o.material){for(const m of [].concat(o.material)){m.map?.dispose();m.dispose();}}});}

function label(str,p,size=.24,color='#767b6d'){
 const c=document.createElement('canvas');let ctx=c.getContext('2d');ctx.font='32px "Microsoft YaHei", sans-serif';c.width=Math.ceil(ctx.measureText(str).width+24);c.height=56;ctx=c.getContext('2d');ctx.font='32px "Microsoft YaHei", sans-serif';ctx.fillStyle=color;ctx.textAlign='center';ctx.fillText(str,c.width/2,38);
 const texture=new THREE.CanvasTexture(c);const s=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,transparent:true,depthTest:false}));s.scale.set(size*1.65*c.width/56,size*1.65,1);s.position.copy(p);return s;
}
function line(points,color,dashed=false,opacity=1){
 const g=new THREE.BufferGeometry().setFromPoints(points.map(world));const material=dashed?new THREE.LineDashedMaterial({color,dashSize:span*.009,gapSize:span*.005,transparent:true,opacity}):new THREE.LineBasicMaterial({color,transparent:true,opacity});
 const l=new THREE.Line(g,material);if(dashed)l.computeLineDistances();return l;
}
function initScene(){
 const box=$('viewport');scene=new THREE.Scene();scene.background=new THREE.Color('#eeeee7');camera=new THREE.PerspectiveCamera(38,1,.001,1000);camera.up.set(0,1,0);
 renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));box.appendChild(renderer.domElement);
 renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label','参考路线与学生实际飞行三维对比');
 controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.09;controls.maxPolarAngle=Math.PI*.93;
 scene.add(new THREE.HemisphereLight(0xffffff,0x778da0,2.4));const light=new THREE.DirectionalLight(0xffffff,2.2);light.position.set(10,20,5);scene.add(light);
 new ResizeObserver(()=>{const w=box.clientWidth,h=box.clientHeight;if(w&&h){camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);}}).observe(box);
}
function setView(name){
 if(!camera)return;const d=span*(camera.aspect<1?2.25:1.85);
 const offset=name==='top'?new THREE.Vector3(0,d,.001):name==='side'?new THREE.Vector3(d,.08*d,0):new THREE.Vector3(1.02,.78,1.1).normalize().multiplyScalar(d);
 camera.position.copy(center).add(offset);controls.target.copy(center);controls.minDistance=span*.03;controls.maxDistance=span*10;camera.near=span/10000;camera.far=span*100;camera.updateProjectionMatrix();controls.update();
}
function showGeometry(){
 if(!scene)return;
 if(group){scene.remove(group);dispose(group);}group=new THREE.Group();scene.add(group);plane=null;
 const all=trace.reference.concat(trace.position);const bounds=new THREE.Box3().setFromPoints(all.map(world));const size=bounds.getSize(new THREE.Vector3());center=bounds.getCenter(new THREE.Vector3());span=Math.max(size.x,size.y,size.z,1);
 group.add(line(trace.reference,'#b77737',true));
 if(trace.position.length){actualLine=line(trace.position,'#426b59');group.add(actualLine);}
 const floor=Math.floor((bounds.min.y-span*.08)*2)/2,step=Math.pow(10,Math.floor(Math.log10(span/5))),tick=span/step>10?step*2:step;
 const lowE=Math.floor((bounds.min.x-span*.12)/tick)*tick,highE=Math.ceil((bounds.max.x+span*.12)/tick)*tick;
 const lowN=Math.floor((-bounds.max.z-span*.12)/tick)*tick,highN=Math.ceil((-bounds.min.z+span*.12)/tick)*tick;
 const pts=[];for(let e=lowE;e<=highE+1e-8;e+=tick){pts.push(new THREE.Vector3(e,floor,-lowN),new THREE.Vector3(e,floor,-highN));group.add(label(fmt(e,1),new THREE.Vector3(e,floor,-lowN+span*.05),span*.024));}
 for(let n=lowN;n<=highN+1e-8;n+=tick){pts.push(new THREE.Vector3(lowE,floor,-n),new THREE.Vector3(highE,floor,-n));group.add(label(fmt(n,1),new THREE.Vector3(highE+span*.055,floor,-n),span*.024));}
 const gg=new THREE.BufferGeometry().setFromPoints(pts);group.add(new THREE.LineSegments(gg,new THREE.LineBasicMaterial({color:'#cad0c2',transparent:true,opacity:.75})));
 group.add(label('东向 / km',new THREE.Vector3((lowE+highE)/2,floor,-lowN+span*.14),span*.035));
 group.add(label('北向 / km',new THREE.Vector3(highE+span*.18,floor,-(lowN+highN)/2),span*.035));
 const vp=[new THREE.Vector3(lowE,floor,-highN),new THREE.Vector3(lowE,bounds.max.y+span*.12,-highN)];group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(vp),new THREE.LineBasicMaterial({color:'#aeb7a4'})));
 const hstep=Math.max(tick/2,.25);for(let h=Math.ceil(floor/hstep)*hstep;h<=bounds.max.y+span*.12;h+=hstep)group.add(label(fmt(h,1),new THREE.Vector3(lowE-span*.05,h,-highN),span*.024));
 group.add(label('高度 / km',new THREE.Vector3(lowE,bounds.max.y+span*.19,-highN),span*.035));
 const start=new THREE.Mesh(new THREE.SphereGeometry(span*.006,16,12),new THREE.MeshBasicMaterial({color:'#399785'}));start.position.copy(world(trace.reference[0]||trace.position[0]));group.add(start);
 if(trace.position.length){
  if(trace.attitude_complete===false){
   plane=new THREE.Mesh(new THREE.SphereGeometry(span*.008,18,14),new THREE.MeshStandardMaterial({color:'#426b59'}));plane.userData.positionOnly=true;group.add(plane);
  }else{
  const model=aircraftMesh({position:[0,0,0],hprv:[0,0,0,250]},span*1000*.08);const vertices=[],colors=[];
  model.i.forEach((_,f)=>{const color=new THREE.Color(model.facecolor[f]);for(const j of [model.i[f],model.j[f],model.k[f]]){vertices.push(model.x[j],model.z[j],-model.y[j]);colors.push(color.r,color.g,color.b);}});
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));g.computeVertexNormals();
  plane=new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:.5,metalness:.2,flatShading:true}));plane.renderOrder=3;group.add(plane);
  }
 }
 setView('perspective');
}
function updatePlane(i){
 if(!plane)return;
 if(plane.userData.positionOnly){plane.position.copy(world(trace.position[i]));return;}
 const m=aircraftMesh({position:[0,0,0],hprv:trace.hprv[i]},span*1000*.08);const a=plane.geometry.getAttribute('position');let k=0;
 m.i.forEach((_,f)=>{for(const j of [m.i[f],m.j[f],m.k[f]])a.setXYZ(k++,m.x[j],m.z[j],-m.y[j]);});a.needsUpdate=true;plane.geometry.computeVertexNormals();plane.position.copy(world(trace.position[i]));
}
function drawChart(){
 const c=$('chart'),rect=c.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio,2);c.width=Math.max(1,rect.width*dpr);c.height=100*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=100;
 if(!trace?.cte?.length){ctx.fillStyle='#79806f';ctx.font='12px sans-serif';ctx.fillText(trace?.position?.length?'此历史记录未保存跟踪误差。':'无飞行记录，不计算跟踪误差。',10,42);return;}
 const max=Math.max(180,Math.ceil(trace.cte.reduce((m,v)=>Math.max(m,v),0)/50)*50),duration=trace.time.at(-1),x=t=>32+t/duration*(w-45),y=v=>80-v/max*69;
 ctx.font='9px sans-serif';ctx.fillStyle='#79806f';ctx.strokeStyle='#dfe1d4';ctx.lineWidth=1;
 for(const v of [0,150,max]){ctx.beginPath();ctx.moveTo(30,y(v));ctx.lineTo(w-5,y(v));ctx.stroke();ctx.fillText(String(v),1,y(v)+3);}
 if(selected?.set!=='custom'&&!selected?.historical){ctx.setLineDash([4,4]);ctx.strokeStyle='#b77737';ctx.beginPath();ctx.moveTo(30,y(150));ctx.lineTo(w-5,y(150));ctx.stroke();ctx.setLineDash([]);}
 ctx.strokeStyle='#426b59';ctx.lineWidth=1.6;ctx.beginPath();trace.cte.forEach((v,i)=>i?ctx.lineTo(x(trace.time[i]),y(v)):ctx.moveTo(x(trace.time[i]),y(v)));ctx.stroke();
 ctx.strokeStyle='#3b453b';ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(x(clock),5);ctx.lineTo(x(clock),83);ctx.stroke();ctx.fillText('0 s',30,97);ctx.fillText(`${fmt(duration)} s`,w-43,97);
}
function drawFrame(){
 if(!trace?.position.length){text('time','—');text('telemetry','');drawChart();return;}
 let i=0;while(i<trace.time.length-1&&trace.time[i+1]<=clock+1e-6)i++;
 if(i!==frame||!plane?.userData.ready){frame=i;updatePlane(i);if(plane)plane.userData.ready=true;}
 $('scrub').value=clock;text('time',`${fmt(clock)} / ${fmt(trace.time.at(-1))} s`);
 const h=trace.hprv[i];$('telemetry').innerHTML=`<div><span>空速</span><b>${fmt(h[3])} m/s</b></div><div><span>滚转</span><b>${fmt(h[2]*180/Math.PI)}°</b></div><div><span>侧滑</span><b>${fmt(trace.beta[i])}°</b></div>`;
 drawChart();
}
function renderList(){
 if(!catalog)return;const set=$('dataset').value,s=$('status').value,q=$('search').value.trim().toLowerCase();text('set-note',catalog.datasets?.[set]?.note||sets[set]||'历史实飞记录');
 const rows=catalog.rows.filter(r=>replayDataset(r)===set);
 const list=rows.filter(r=>(s==='all'||r.status===s)&&(`${r.index}`.padStart(3,'0')+' '+r.name+' '+(r.modelLabel||'')+' '+(r.recordedDate||'')).toLowerCase().includes(q));
 text('list-count',`显示 ${list.length} / ${catalog.counts[set]?.requested??rows.length} 项`);
 $('route-list').replaceChildren();
 for(const r of list){const b=document.createElement('button');b.className='route'+(selected?.id===r.id?' selected':'');b.dataset.id=r.id;b.setAttribute('aria-pressed',String(selected?.id===r.id));
  const name=document.createElement('span');name.className='route-name';name.textContent=`${String(r.index).padStart(3,'0')} ${r.name}${r.condition?' · 初态扰动':''}`;
  const meta=document.createElement('span');meta.className='route-meta';meta.innerHTML=`<span>${r.set==='custom'?'实际航迹':`<i class="status-dot ${r.status}"></i>${statusText[r.status]}`}</span><span>${r.flown?(Number.isFinite(r.cteP90)?'CTE P90 '+fmt(r.cteP90)+' m':fmt(r.seconds)+' s 实飞'):'不虚构飞行轨迹'}</span>`;b.append(name,meta);if(r.historical){const source=document.createElement('small');source.className='route-source';source.textContent=(r.modelLabel||'历史学生')+' · '+(r.recordedDate||'早期归档');b.append(source);}b.onclick=()=>selectCase(r.id);$('route-list').append(b);}
 if(!list.length){const p=document.createElement('p');p.className='muted';p.textContent='没有匹配项，请修改筛选条件。';$('route-list').append(p);}
 return list;
}
async function selectCase(id){
 const n=++request;selected=catalog.rows.find(r=>r.id===id);playing=false;text('play','▶ 播放');renderList();const r=selected;
 text('case-title',`${r.set==='custom'?'':String(r.index).padStart(3,'0')+' '}${r.name}${r.condition?' · 初态扰动':''}`);text('case-badge',statusText[r.status]);$('case-badge').className='badge '+r.status;text('cp',r.modelLabel||`${['keeper','custom'].includes(r.set)?'保留模型':'未晋级候选'} / ${r.checkpoint.slice(0,8)}`);
 const custom=r.set==='custom';$('case-badge').hidden=custom;if(custom)text('case-badge','');
 document.querySelector('.chart-label span:last-child').hidden=custom||r.historical;
 text('case-note',r.historical?`${r.modelLabel} · ${r.note}`:custom?(r.note||'实际飞行记录'):(r.reasons.length?'未达标项：'+r.reasons.join('、'):'本次飞行通过'));
 for(const [element,path] of [['download-acmi',r.acmi],['download-plot',r.plot]]){$(element).hidden=!path;if(path)$(element).href=path;}
 $('download-acmi').download=`飞行记录_${r.id}.acmi`;
 $('metrics').innerHTML=[['CTE P90',r.cteP90,'m'],['最大偏离',r.cteMax,'m'],['终点距离',r.endpoint,'m'],custom||r.historical?['实际飞行时长',r.seconds,'s']:['安全 / 非指令整圈滚转',r.flown?`${r.safety?'✓':'×'} / ${r.spin?'✓':'×'}`:'—','']].map(([t,v,u])=>`<div class="metric"><span>${t}</span><strong>${typeof v==='string'?v:fmt(v)}</strong><em>${u}</em></div>`).join('');
 try{const response=await fetch(r.data);if(!response.ok)throw Error('记录文件加载失败');const data=await response.json();if(n!==request)return;trace=data;clock=trace.time[0]||0;frame=-1;$('scrub').min=clock;$('scrub').max=trace.time.at(-1)||1;$('scrub').disabled=!r.flown;$('play').disabled=!r.flown;$('no-flight').hidden=r.flown;showGeometry();drawFrame();location.hash=`flight=${r.id}`;window.__flightReady={id:r.id,frames:trace.position.length,reference:trace.reference.length,attitudeComplete:trace.attitude_complete!==false};}
 catch(error){text('case-note','加载失败：'+error.message);window.__flightError=error.message;}
}
function evidenceTable(){
 const groups=new Map();for(const r of catalog.rows.filter(x=>x.set==='keeper')){if(!groups.has(r.name))groups.set(r.name,{total:0,pass:0,fail:0,refused:0});const g=groups.get(r.name);g.total++;g[r.status]++;}
 let html='<table><thead><tr><th>路线类别</th><th>通过</th><th>失败</th><th>未飞</th></tr></thead><tbody>';
 for(const [name,g] of groups)html+=`<tr><td>${name}</td><td>${g.pass}/${g.total}</td><td>${g.fail}</td><td>${g.refused}</td></tr>`;
 $('family-table').innerHTML=html+'</tbody></table>';
}
$('dataset').onchange=()=>{$('status').value='all';$('search').value='';const list=renderList();if(list.length)selectCase(list[0].id);};$('status').onchange=renderList;$('search').oninput=renderList;
$('play').onclick=()=>{if(!trace?.time.length)return;if(clock>=trace.time.at(-1))clock=trace.time[0]||0;playing=!playing;text('play',playing?'Ⅱ 暂停':'▶ 播放');};
$('scrub').oninput=e=>{clock=Number(e.target.value);drawFrame();};
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));
$('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.querySelector('.viewport-wrap').requestFullscreen();}catch{ text('fullscreen','浏览器不支持'); }};
$('chart').onclick=e=>{if(!trace?.time.length)return;const r=e.target.getBoundingClientRect();clock=Math.max(0,Math.min(1,(e.clientX-r.left-32)/(r.width-45)))*trace.time.at(-1);drawFrame();};
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!['INPUT','SELECT','BUTTON','TEXTAREA'].includes(e.target.tagName)){e.preventDefault();$('play').click();}});
window.addEventListener('resize',drawChart);
function tick(now){const dt=Math.min((now-last)/1000,.1);last=now;if(playing&&trace?.time.length){clock=Math.min(clock+dt*Number($('speed').value),trace.time.at(-1));drawFrame();if(clock>=trace.time.at(-1)){playing=false;text('play','↻ 重播');}}if(renderer){controls.update();renderer.render(scene,camera);}requestAnimationFrame(tick);}
try{initScene();}catch(error){$('render-error').hidden=false;text('render-error','此浏览器无法创建 3D 视图。请开启硬件加速或使用 Chrome / Edge；原始图表与 ACMI 仍可下载。');}
requestAnimationFrame(tick);
window.showCustomFlight=async function(result){
 const r={...result.meta,seconds:result.meta.seconds??result.trace.time.at(-1),id:'custom_'+result.id,set:'custom',index:0,condition:0,role:'本次真实试飞',
 data:URL.createObjectURL(new Blob([JSON.stringify(result.trace)],{type:'application/json'})),
 acmi:result.acmi?URL.createObjectURL(new Blob([acmiDisplayModels(result.acmi)],{type:'text/plain'})):null,plot:null};
 catalog={rows:[r],counts:{custom:{requested:1}}};$('dataset').replaceChildren(new Option('自定义试飞','custom'));await selectCase(r.id);
};
try{if(!document.body.dataset.custom){const response=await fetch('catalog.json?v=12');if(!response.ok)throw Error('catalog');catalog=await response.json();const historical=catalog.rows.filter(r=>r.historical);if(historical.length)$('dataset').append(new Option(`特技机动 · ${new Set(historical.map(r=>r.set)).size}类、${historical.length}条飞行记录`,'aerobatics'));const deep=location.hash.startsWith('#flight=')?catalog.rows.find(r=>r.id===location.hash.slice(8)):null;if(deep)$('dataset').value=replayDataset(deep);renderList();await selectCase(deep?.id||'new_077_0');}}
catch(error){text('case-title','数据未能加载，请通过网页服务器打开');text('case-note','离线使用请参阅部署说明，不能直接双击 index.html。');window.__flightError=error.message;}
