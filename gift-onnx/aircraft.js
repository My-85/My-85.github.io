// J-20-inspired code-native display mesh: long chined nose, canards, delta wings,
// two engines and outward-canted twin fins. Display only; not an aerodynamic model.
// Body x forward, y right, z down; attitude always comes from the real trace.
export function acmiDisplayModels(text){
 // Also restyle previously saved jobs when re-downloaded, without touching T= or time.
 return text.split('\n').map(line=>{
  if(!line.includes(',Type=Air+FixedWing'))return line;
  const ref=/reference|ghost|参考/i.test(line);
  return line.replace(/,Name=[^,\r]*/,`,Name=${ref?'Su-57':'J-20'}`)
   .replace(/,CallSign=[^,\r]*/,'')+`,CallSign=${ref?'Reference (geometry only)':'Student RL (actual flight)'}`;
 }).join('\n');
}
export function bodyToNEU(body,hprv){
 const [yaw,pitch,roll]=hprv,[x,y,z]=body;
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),cr=Math.cos(roll),sr=Math.sin(roll);
 const n=cy*cp*x+(cy*sp*sr-sy*cr)*y+(cy*sp*cr+sy*sr)*z;
 const e=sy*cp*x+(sy*sp*sr+cy*cr)*y+(sy*sp*cr-cy*sr)*z;
 const down=-sp*x+cp*sr*y+cp*cr*z;
 return [n,e,-down];
}
export function aircraftMesh(sample,lengthM){
 if(!sample?.position||!sample?.hprv||!sample.position.concat(sample.hprv.slice(0,3)).every(Number.isFinite))return null;
 const vertices=[],faces=[],colors=[];
 const add=(ps,tri,color)=>{const start=vertices.length;vertices.push(...ps);for(const f of tri){faces.push(f.map(i=>i+start));colors.push(color);}};
 const tri=(a,b,c,color)=>add([a,b,c],[[0,1,2]],color);
 const quad=(a,b,c,d,color)=>add([a,b,c,d],[[0,1,2],[0,2,3]],color);
 const skin='#91a7b5',upper='#aabec9',dark='#405f75',edge='#64869e';
 const rings=[[1.2,.008,.005],[.72,.07,.058],[.22,.17,.088],[-.38,.24,.09],[-.88,.205,.073]];
 for(let j=0;j<rings.length-1;j++)for(let k=0;k<8;k++){
  const v=(r,a)=>[rings[r][0],rings[r][1]*Math.cos(a*Math.PI/4),rings[r][2]*Math.sin(a*Math.PI/4)];
  quad(v(j,k),v(j,k+1),v(j+1,k+1),v(j+1,k),k>=4?upper:skin);
 }
 for(const sign of [-1,1]){
  // Swept chines and broad cropped-delta main wing, with no rear horizontal tail.
  quad([.73,sign*.065,0],[.13,sign*.235,.003],[-.76,sign*.27,.003],[-.86,sign*.18,.008],upper);
  quad([.14,sign*.15,.012],[-.49,sign*.69,.015],[-.73,sign*.69,.018],[-.76,sign*.18,.016],skin);
  tri([-.49,sign*.69,.013],[-.73,sign*.69,.016],[-.67,sign*.54,.012],edge);
  // Distinct forward canards.
  quad([.52,sign*.115,-.023],[.25,sign*.405,-.02],[.08,sign*.405,-.018],[.15,sign*.14,-.018],upper);
  // Dark side inlet face and its raised shoulder.
  quad([.23,sign*.18,-.052],[.23,sign*.23,-.026],[.23,sign*.23,.065],[.23,sign*.14,.07],dark);
  quad([.23,sign*.23,-.026],[-.21,sign*.28,-.05],[-.63,sign*.22,-.05],[.23,sign*.18,-.052],edge);
  // Outward-canted twin vertical fins.
  quad([-.41,sign*.17,-.06],[-.66,sign*.30,-.38],[-.88,sign*.30,-.36],[-.91,sign*.19,-.045],edge);
  // Two distinct exhaust nozzles, not a single center engine.
  const center=sign*.115,r=.077;
  for(let k=0;k<12;k++){
   const v=(x,a,rr)=>[x,center+rr*Math.cos(a*Math.PI/6),.004+rr*Math.sin(a*Math.PI/6)];
   quad(v(-.62,k,r),v(-.62,k+1,r),v(-1.04,k+1,r*.84),v(-1.04,k,r*.84),k>=6?edge:dark);
   tri([-1.045,center,.004],v(-1.045,k,r*.77),v(-1.045,k+1,r*.77),'#233a49');
  }
 }
 add([[.70,-.044,-.042],[.70,.044,-.042],[.31,.089,-.072],[.31,-.089,-.072],[.48,0,-.165]],
  [[0,1,4],[1,2,4],[2,3,4],[3,0,4]],'#1f485c');
 const neu=vertices.map(v=>bodyToNEU(v.map(x=>x*lengthM/2.24),sample.hprv).map((x,k)=>x+sample.position[k]));
 return {type:'mesh3d',name:'学生飞机（实际姿态，外形放大）',x:neu.map(p=>p[1]/1000),y:neu.map(p=>p[0]/1000),z:neu.map(p=>p[2]/1000),
  i:faces.map(f=>f[0]),j:faces.map(f=>f[1]),k:faces.map(f=>f[2]),facecolor:colors,
  flatshading:true,showscale:false,showlegend:false,hoverinfo:'name',
  meta:{real_position_neu_m:sample.position.slice(),real_hprv:sample.hprv.slice(),display_length_m:lengthM,
   display_shape:'J-20',source:'actual recorded state; enlarged illustrative shape'}};
}
