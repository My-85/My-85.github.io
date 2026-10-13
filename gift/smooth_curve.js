// Local chord-parameter Hermite interpolation: linear cost, every control
// retained, including vertical/reversing/closed paths and coincident controls.
export function sampleCurve(points, samplesPerSegment=40){
 if(!Array.isArray(points)||points.length<2||points.some(p=>!Array.isArray(p)||p.length!==3||p.some(v=>!Number.isFinite(v))))throw Error('需要至少两个有限三维控制点');
 const knots=[];
 points.forEach((p,i)=>{if(!knots.length||p.some((v,k)=>v!==knots.at(-1).p[k]))knots.push({p,index:i});else knots.at(-1).index=i;});
 if(knots.length<2)throw Error('路线没有长度，请移动或添加一个点');
 const distance=(a,b)=>Math.hypot(...a.map((v,k)=>v-b[k]));
 const slopes=knots.map(({p},i)=>{
  if(i===0)return knots[1].p.map((v,k)=>(v-p[k])/distance(p,knots[1].p));
  if(i===knots.length-1)return p.map((v,k)=>(v-knots[i-1].p[k])/distance(p,knots[i-1].p));
  const a=knots[i-1].p,b=knots[i+1].p,d=distance(a,p)+distance(p,b);
  return b.map((v,k)=>(v-a[k])/d);
 });
 const positions=[],controlSegments=[];
 for(let i=0;i<knots.length-1;i++){
  const a=knots[i].p,b=knots[i+1].p,d=distance(a,b);
  for(let j=0;j<samplesPerSegment;j++){
   const t=j/samplesPerSegment,t2=t*t,t3=t2*t;
   const p=a.map((v,k)=>(2*t3-3*t2+1)*v+(t3-2*t2+t)*d*slopes[i][k]+(-2*t3+3*t2)*b[k]+(t3-t2)*d*slopes[i+1][k]);
   if(!positions.length||p.some((v,k)=>v!==positions.at(-1)[k])){positions.push(p);controlSegments.push(Math.min(knots[i].index,points.length-2));}
  }
 }
 positions.push(knots.at(-1).p.slice());controlSegments.push(Math.min(knots.at(-2).index,points.length-2));
 return {positions,controlSegments};
}
export function smoothCurve(points){return sampleCurve(points).positions;}
