const VERSION='gift-browser-pages-20261015-v1', ASSETS='gift-browser-assets-20261015-v1';
const root=new URL('./',self.location.href);
self.addEventListener('install',event=>event.waitUntil((async()=>{
  const response=await fetch(new URL('offline-files.json',root));
  if(!response.ok)throw Error('Offline manifest missing');
  const files=await response.json(),cache=await caches.open(VERSION);
  // Core editor/replay assets fit comfortably into the first page cache.
  await cache.addAll(files.map(path=>new URL(path,root).href));
  await self.skipWaiting();
})()));
self.addEventListener('activate',event=>event.waitUntil(self.clients.claim()));
self.addEventListener('fetch',event=>{
  const request=event.request,url=new URL(request.url);
  if(request.method!=='GET')return;
  const shared=url.origin==='https://my-85.github.io'&&url.pathname.startsWith('/gift-browser/runtime/');
  if(!shared&&(url.origin!==root.origin||(!url.pathname.startsWith(root.pathname)&&!url.pathname.startsWith('/gift-browser/runtime/'))))return;
  event.respondWith((async()=>{
    const cache=await caches.open(VERSION),assets=await caches.open(ASSETS);
    const old=await cache.match(request)||await assets.match(request);
    if(old)return old;
    try {
      const response=await fetch(request);
      if(response.ok&&response.type!=='opaque')await cache.put(request,response.clone());
      return response;
    } catch(error) {
      if(request.mode==='navigate'){
        const fallback=await cache.match(new URL('custom.html',root));
        if(fallback)return fallback;
      }
      throw error;
    }
  })());
});
