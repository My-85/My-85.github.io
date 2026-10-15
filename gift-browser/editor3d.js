import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {TransformControls} from './vendor/TransformControls.js';
import {nextPointSector} from './drawing_guard.js';

// NEU metres <-> right-handed display: east, altitude, minus north in km.
const vec=p=>new THREE.Vector3(p[1]/1000,p[2]/1000,-p[0]/1000);
const neu=v=>[-v.z*1000,v.x*1000,v.y*1000].map(Math.round);
export class SpatialEditor {
  constructor(container,callbacks){
    this.box=container;this.cb=callbacks;this.points=[];this.selected=1;this.editable=false;
    this.mode='select';this.planeName='NE';this.planeValue=5000;this.drag=null;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#f4f7fa');
    this.camera=new THREE.PerspectiveCamera(42,1,.01,1200);
    this.camera.position.set(17,20,14);
    this.renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    container.appendChild(this.renderer.domElement);
    this.orbit=new OrbitControls(this.camera,this.renderer.domElement);
    this.orbit.target.set(2,5,-4);this.orbit.enableDamping=false;this.orbit.update();
    this.transform=new TransformControls(this.camera,this.renderer.domElement);
    this.transform.setMode('translate');this.transform.setSize(.9);this.transform.setTranslationSnap(.025);
    this.scene.add(this.transform.getHelper());
    this.transform.addEventListener('dragging-changed',e=>{
      this.orbit.enabled=!e.value;if(e.value){this.cb.begin();this.gizmoMoved=true;}else if(this.gizmoMoved){this.cb.commit();this.gizmoMoved=false;}
    });
    this.transform.addEventListener('objectChange',()=>{
      if(this.transform.object&&this.selected>=0){this.cb.move(this.selected,neu(this.transform.object.position));this.render();}
    });
    this.markers=[];this.labels=[];this.lines=new THREE.Group();this.scene.add(this.lines);
    this.axes=new THREE.Group();this.scene.add(this.axes);
    this.issueFocus=new THREE.Group();this.scene.add(this.issueFocus);
    this.guards=new THREE.Group();this.scene.add(this.guards);this.guard=null;
    this.ray=new THREE.Raycaster();
    this.drawingPlane=new THREE.Plane();
    this.planeMesh=new THREE.Mesh(new THREE.PlaneGeometry(30,30),new THREE.MeshBasicMaterial({color:'#9cab92',opacity:.10,transparent:true,side:THREE.DoubleSide,depthWrite:false}));
    this.scene.add(this.planeMesh);this.updatePlane();
    const canvas=this.renderer.domElement;canvas.setAttribute('aria-label','三维路线编辑器');canvas.tabIndex=0;
    canvas.addEventListener('pointerdown',e=>this.down(e),true);
    canvas.addEventListener('pointermove',e=>this.move(e),true);
    canvas.addEventListener('pointerup',e=>this.up(e),true);
    canvas.addEventListener('pointercancel',e=>this.up(e),true);
    this.orbit.addEventListener('change',()=>this.render());
    this.transform.addEventListener('change',()=>this.render());
    new ResizeObserver(()=>this.resize()).observe(container);
    this.resize();this.setBounds([[0,0,5000],[12000,8000,6500]]);
  }
  resize(){const w=this.box.clientWidth,h=this.box.clientHeight;if(!w||!h)return;this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.renderer.setSize(w,h);this.render();}
  render(){this.camera.updateMatrixWorld();for(const m of this.markers){const d=this.camera.position.distanceTo(m.position);m.scale.setScalar(d*.008);}
    const occupied=[];this.labels.forEach((label,i)=>{const v=this.markers[i].position.clone().project(this.camera),screen=[v.x*this.box.clientWidth/2,v.y*this.box.clientHeight/2];label.visible=i===this.selected||!occupied.some(p=>Math.hypot(p[0]-screen[0],p[1]-screen[1])<28);if(label.visible)occupied.push(screen);});for(const s of this.labels.concat(this.axes.children.filter(c=>c.isSprite),this.issueFocus.children.filter(c=>c.isSprite))){const d=this.camera.position.distanceTo(s.position);s.scale.set(d*(s.userData.compact?.045:.13),d*.022,1);}this.renderer.render(this.scene,this.camera);}
  project(p){const v=vec(p).project(this.camera),r=this.renderer.domElement.getBoundingClientRect();return {x:r.left+(v.x+1)*r.width/2,y:r.top+(1-v.y)*r.height/2,z:v.z};}
  nearest(e){let best=-1,distance=18;this.points.forEach((p,i)=>{const v=this.project(p),d=Math.hypot(v.x-e.clientX,v.y-e.clientY);if(v.z>-1&&v.z<1&&d<distance){best=i;distance=d;}});return best;}
  intersection(e,plane=this.drawingPlane){const r=this.renderer.domElement.getBoundingClientRect();this.ray.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2),this.camera);return this.ray.ray.intersectPlane(plane,new THREE.Vector3());}
  down(e){
    if(e.button!==0||!this.editable||this.transform.axis)return;
    this.press={x:e.clientX,y:e.clientY};const index=this.nearest(e);
    if(index>=0){
      e.stopImmediatePropagation();this.cb.select(index);this.select(index);
      if(index>=0){
        this.cb.begin();this.orbit.enabled=false;
        const normal=e.shiftKey?this.camera.getWorldDirection(new THREE.Vector3()):this.drawingPlane.normal.clone();
        const plane=new THREE.Plane().setFromNormalAndCoplanarPoint(normal,vec(this.points[index]));
        const hit=this.intersection(e,plane);
        this.drag={index,plane,offset:hit?vec(this.points[index]).sub(hit):new THREE.Vector3(),moved:false};
        this.renderer.domElement.setPointerCapture(e.pointerId);
      }
    }else if(this.mode==='add'){
      this.orbit.enabled=false;this.addPress=true;e.stopImmediatePropagation();this.renderer.domElement.setPointerCapture(e.pointerId);
    }
  }
  move(e){if(!this.drag)return;e.stopImmediatePropagation();
    if(Math.hypot(e.clientX-this.press.x,e.clientY-this.press.y)<3&&!this.drag.moved)return;
    const hit=this.intersection(e,this.drag.plane);if(hit){this.drag.moved=true;this.cb.move(this.drag.index,neu(hit.add(this.drag.offset)));}
  }
  up(e){
    if(this.drag){e.stopImmediatePropagation();if(this.drag.moved)this.cb.commit();else this.cb.abort();this.drag=null;this.orbit.enabled=true;}
    if(this.addPress){e.stopImmediatePropagation();this.addPress=false;this.orbit.enabled=true;
      if(Math.hypot(e.clientX-this.press.x,e.clientY-this.press.y)<5){const p=this.intersection(e);if(p)this.cb.add(neu(p));else this.cb.message('当前视角与画布平行，请稍微旋转视角。');}
    }
  }
  select(index){this.selected=index;this.markers.forEach((m,i)=>m.material.color.set(i===index?'#d98b32':'#007eb8'));
    if(this.editable&&index>=0&&this.markers[index])this.transform.attach(this.markers[index]);else this.transform.detach();this.render();}
  setMode(mode){this.mode=mode;this.updatePlane();}
  setPlane(name,value){this.planeName=name;this.planeValue=value;this.updatePlane();}
  updatePlane(){
    const val=this.planeValue/1000;let normal,at;
    if(this.planeName==='NE'){normal=new THREE.Vector3(0,1,0);at=new THREE.Vector3(0,val,0);}
    else if(this.planeName==='NH'){normal=new THREE.Vector3(1,0,0);at=new THREE.Vector3(val,5,-5);}
    else{normal=new THREE.Vector3(0,0,1);at=new THREE.Vector3(0,5,-val);}
    this.drawingPlane.setFromNormalAndCoplanarPoint(normal,at);this.planeMesh.position.copy(at);
    this.planeMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),normal);
    this.planeMesh.visible=this.mode==='add';this.updateGuard();this.render();
  }
  textSprite(text,color='#677261',compact=false){
    const c=document.createElement('canvas');c.width=compact?128:384;c.height=64;const t=c.getContext('2d');t.font=(compact?'bold 40px':'36px')+' sans-serif';t.textAlign='center';t.fillStyle=color;t.fillText(text,c.width/2,44);
    const texture=new THREE.CanvasTexture(c),s=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false,transparent:true}));s.userData.compact=compact;s.scale.set(2.4,.4,1);s.renderOrder=15;return s;
  }
  clear(group){while(group.children.length){const c=group.children[0];group.remove(c);c.geometry?.dispose();c.material?.map?.dispose();c.material?.dispose();}}
  line(ps,color,group=this.lines){
    let run=[];const flush=()=>{if(run.length>1){const g=new THREE.BufferGeometry().setFromPoints(run);group.add(new THREE.Line(g,new THREE.LineBasicMaterial({color})));}run=[];};
    ps.forEach(p=>{if(p)run.push(vec(p));else flush();});flush();
  }
  update(points,selected,reference=[],risk=[],actual=[],editable=true){
    this.points=points;this.editable=editable;
    while(this.markers.length>points.length){const m=this.markers.pop();this.scene.remove(m);m.geometry.dispose();m.material.dispose();const label=this.labels.pop();this.scene.remove(label);label.material.map.dispose();label.material.dispose();}
    points.forEach((p,i)=>{
      if(!this.markers[i]){const m=new THREE.Mesh(new THREE.SphereGeometry(1,16,12),new THREE.MeshBasicMaterial({color:'#007eb8',depthTest:false}));m.renderOrder=10;this.scene.add(m);this.markers.push(m);const label=this.textSprite(String(i),'#526a57',true);this.labels.push(label);this.scene.add(label);}
      this.markers[i].position.copy(vec(p));this.markers[i].visible=true;this.labels[i].position.copy(vec(p)).add(new THREE.Vector3(.2,.35,0));
    });
    this.clear(this.lines);this.line(points,'#b7c1ae');this.line(reference,'#d98b32');this.line(risk,'#a94d3d');this.line(actual,'#007eb8');this.updateGuard();this.select(selected);
  }
  setGuard(g){this.guard=g;this.updateGuard();this.render();}
  updateGuard(){
    this.clear(this.guards);const g=this.guard;if(!g||!this.points.length)return;
    const n=g.max_north_m,e=(n-1000)*Math.tan(g.max_heading_from_north_deg*Math.PI/180),low=g.min_altitude_m,high=g.max_altitude_m;
    const outline=[[1000,0],[n,-e],[n,e],[1000,0]];
    for(const h of [low,high])this.line(outline.map(p=>[...p,h]),'#8dbbb1',this.guards);
    for(const p of outline.slice(0,3))this.line([[...p,low],[...p,high]],'#8dbbb1',this.guards);
    const last=this.points.at(-1),ring=nextPointSector(last,g);
    // Clip the horizontal sector at the north budget. This is a placement guide,
    // not a claim about nonlinear spline feasibility or student success.
    const clipped=[];
    for(let i=0;i<ring.length;i++){
      const a=ring[i],b=ring[(i+1)%ring.length],inside=a[0]<=n,next=b[0]<=n;
      if(inside)clipped.push(a);
      if(inside!==next){const t=(n-a[0])/(b[0]-a[0]);clipped.push(a.map((v,k)=>v+t*(b[k]-v)));}
    }
    if(clipped.length>2){
      const h=this.planeName==='NE'?this.planeValue:last[2];
      for(const altitude of [low,high])this.line(clipped.concat([clipped[0]]).map(p=>[p[0],p[1],altitude]),'#6baa90',this.guards);
      const shape=new THREE.Shape(clipped.map(p=>new THREE.Vector2(p[1]/1000,-p[0]/1000)));
      const mesh=new THREE.Mesh(new THREE.ShapeGeometry(shape),new THREE.MeshBasicMaterial({color:'#64b99a',opacity:.22,transparent:true,side:THREE.DoubleSide,depthWrite:false}));
      mesh.rotation.x=Math.PI/2;mesh.position.y=Math.max(low,Math.min(high,h))/1000;
      this.guards.add(mesh);
    }
    const label=this.textSprite('默认高度 4.6—5.8 km · 绿色仅为放置范围','#35705c');
    label.position.copy(vec([1000,0,high+180]));this.guards.add(label);
  }
  setBounds(ps){
    const p=ps.length?ps:[[0,0,5000],[10000,5000,6000]],b=new THREE.Box3().setFromPoints(p.map(vec));
    const center=b.getCenter(new THREE.Vector3()),size=b.getSize(new THREE.Vector3());
    size.x=Math.max(6,size.x*1.2);size.z=Math.max(6,size.z*1.2);size.y=Math.max(3,size.y*1.2);
    const lo=center.clone().sub(size.clone().multiplyScalar(.5)),hi=center.clone().add(size.clone().multiplyScalar(.5));
    this.clear(this.axes);
    for(let i=0;i<=5;i++){
      const x=lo.x+(hi.x-lo.x)*i/5,z=lo.z+(hi.z-lo.z)*i/5,y=lo.y+(hi.y-lo.y)*i/5;
      const edge=(a,b)=>this.line([neu(a),neu(b)],'#cad0c2',this.axes);
      edge(new THREE.Vector3(x,lo.y,lo.z),new THREE.Vector3(x,lo.y,hi.z));
      edge(new THREE.Vector3(lo.x,lo.y,z),new THREE.Vector3(hi.x,lo.y,z));
      edge(new THREE.Vector3(lo.x,y,lo.z),new THREE.Vector3(lo.x,y,hi.z));
      for(const[text,pos]of[[`东 ${x.toFixed(1)} km`,new THREE.Vector3(x,lo.y,hi.z+.5)],[`北 ${(-z).toFixed(1)} km`,new THREE.Vector3(lo.x-.9,lo.y,z)],[`高 ${y.toFixed(1)} km`,new THREE.Vector3(lo.x-.9,y,lo.z)]]){const t=this.textSprite(text);t.position.copy(pos);this.axes.add(t);}
    }
    this.orbit.target.copy(center);const r=Math.max(size.x,size.y,size.z);
    this.camera.far=Math.max(1200,r*20);this.camera.near=Math.max(.00001,r/100000);this.camera.updateProjectionMatrix();
    this.planeMesh.scale.setScalar(Math.max(1,r/30));
    this.camera.position.copy(center).add(new THREE.Vector3(r*.8,r*.8,r*.95));this.orbit.update();this.render();
  }
  clearIssueFocus(){this.clear(this.issueFocus);this.render();}
  focusIssue(p,section=[]){
    this.clear(this.issueFocus);const at=vec(p);
    const size=section.length?new THREE.Box3().setFromPoints(section.map(vec)).getSize(new THREE.Vector3()).length():3;
    const offset=this.camera.position.clone().sub(this.orbit.target);offset.setLength(Math.min(offset.length(),Math.max(3,size*2.5)));
    this.orbit.target.copy(at);this.camera.position.copy(at).add(offset);this.orbit.update();
    const marker=new THREE.Mesh(new THREE.SphereGeometry(1,20,16),new THREE.MeshBasicMaterial({color:'#a94d3d',depthTest:false,transparent:true,opacity:.85}));
    marker.position.copy(at);marker.scale.setScalar(Math.max(.025,this.camera.position.distanceTo(at)*.012));marker.renderOrder=20;
    this.issueFocus.add(marker);const label=this.textSprite('定位路段','#a94d3d');label.position.copy(at).add(new THREE.Vector3(0,marker.scale.x*3,0));this.issueFocus.add(label);this.focus(p);
  }
  focus(p){const v=vec(p),delta=v.clone().sub(this.orbit.target);this.camera.position.add(delta);this.orbit.target.copy(v);this.orbit.update();this.render();}
  view(name){const r=Math.max(8,this.camera.position.distanceTo(this.orbit.target));const offsets={top:[0,r,.001],north:[r,0,0],east:[0,0,r],perspective:[r*.55,r*.6,r*.65]};this.camera.position.copy(this.orbit.target).add(new THREE.Vector3(...offsets[name]));this.orbit.update();this.render();}
}
