// Mirrors server control-point guards; full spline admission ALWAYS stays server-side.
export function pointIssues(points,g){
  if(!g)return [];
  const errors=[];
  points.forEach((p,i)=>{
    if(i<2)return;
    if(p[0]<g.min_north_m||p[0]>g.max_north_m)errors.push(`点 ${i} 北向需在 0～20 km`);
    if(p[2]<g.min_altitude_m||p[2]>g.max_altitude_m)errors.push(`点 ${i} 高度需在 4600～5800 m`);
    const d=p.map((x,k)=>x-points[i-1][k]),horizontal=Math.hypot(d[0],d[1]);
    if(horizontal<g.min_control_spacing_m||horizontal>g.max_control_spacing_m)errors.push(`点 ${i-1}→${i} 水平距离需为 1.5～6 km`);
    if(Math.abs(Math.atan2(d[1],d[0])*180/Math.PI)>g.max_heading_from_north_deg)errors.push(`点 ${i-1}→${i} 需向北推进，方向在 ±30° 内`);
  });
  return errors;
}

export function nextPointSector(point,g){
  if(!g)return [];
  const ring=[];
  for(const [r,direction] of [[g.max_control_spacing_m,1],[g.min_control_spacing_m,-1]]){
    for(let i=0;i<=24;i++){
      const a=(direction===1?-1+2*i/24:1-2*i/24)*g.max_heading_from_north_deg*Math.PI/180;
      ring.push([point[0]+r*Math.cos(a),point[1]+r*Math.sin(a),point[2]]);
    }
  }
  const clipped=[];
  for(let i=0;i<ring.length;i++){
    const a=ring[i],b=ring[(i+1)%ring.length],inside=a[0]<=g.max_north_m,next=b[0]<=g.max_north_m;
    if(inside)clipped.push(a);
    if(inside!==next){const t=(g.max_north_m-a[0])/(b[0]-a[0]);clipped.push(a.map((v,k)=>v+t*(b[k]-v)));}
  }
  return clipped;
}
