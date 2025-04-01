"use strict";(self.webpackChunkedwardzjl_github_io=self.webpackChunkedwardzjl_github_io||[]).push([[2823],{384:e=>{e.exports=JSON.parse('{"permalink":"/en/blog/static-service","editUrl":"https://github.com/edwardzjl/edwardzjl.github.io/blob/main/blog/2019-07-04-static-service/index.md","source":"@site/blog/2019-07-04-static-service/index.md","title":"\u7cfb\u7edf\u4e2d\u72b6\u6001\u4e3a static \u7684\u670d\u52a1","description":"\u6700\u8fd1\u5f00\u59cb\u63a5\u89e6 Linux \u8fd0\u7ef4\u7684\u5de5\u4f5c\uff0c\u7b2c\u4e00\u4ef6\u4e8b\u60c5\u5c31\u662f\u770b\u770b\u7cfb\u7edf\u4e2d\u8dd1\u4e86\u591a\u5c11\u670d\u52a1\u3002","date":"2019-07-04T00:00:00.000Z","tags":[{"inline":true,"label":"linux","permalink":"/en/blog/tags/linux"}],"readingTime":0.87,"hasTruncateMarker":false,"authors":[{"name":"Junlin Zhou","title":"Fullstack Engineer @ ZJU ICI","url":"https://github.com/edwardzjl","imageURL":"https://github.com/edwardzjl.png","key":"jlzhou","page":null}],"frontMatter":{"slug":"static-service","title":"\u7cfb\u7edf\u4e2d\u72b6\u6001\u4e3a static \u7684\u670d\u52a1","authors":["jlzhou"],"tags":["linux"]},"unlisted":false,"prevItem":{"title":"<\u8bd1>JSON\u683c\u5f0f\u4f5c\u4e3a\u914d\u7f6e\u6587\u4ef6\u7684\u7f3a\u70b9","permalink":"/en/blog/the-downsides-of-json-for-config-files"},"nextItem":{"title":"<\u8bd1> javax.persistence.Id \u548c org.springframework.data.annotation.Id \u7684\u533a\u522b","permalink":"/en/blog/difference-between-javax.persistence.id-and-org.springframework.data.annotation.id"}}')},3120:(e,t,n)=>{n.d(t,{A:()=>i});const i=n.p+"assets/images/services-1f32b5744640cabbd42f360a89b1bffb.png"},5465:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>r,default:()=>u,frontMatter:()=>a,metadata:()=>i,toc:()=>o});var i=n(384),s=n(4848),l=n(8453);const a={slug:"static-service",title:"\u7cfb\u7edf\u4e2d\u72b6\u6001\u4e3a static \u7684\u670d\u52a1",authors:["jlzhou"],tags:["linux"]},r=void 0,c={authorsImageUrls:[void 0]},o=[];function d(e){const t={a:"a",blockquote:"blockquote",code:"code",img:"img",li:"li",p:"p",ul:"ul",...(0,l.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(t.p,{children:"\u6700\u8fd1\u5f00\u59cb\u63a5\u89e6 Linux \u8fd0\u7ef4\u7684\u5de5\u4f5c\uff0c\u7b2c\u4e00\u4ef6\u4e8b\u60c5\u5c31\u662f\u770b\u770b\u7cfb\u7edf\u4e2d\u8dd1\u4e86\u591a\u5c11\u670d\u52a1\u3002"}),"\n",(0,s.jsxs)(t.p,{children:["\u96c6\u7fa4\u7528\u7684\u662f CentOS 7\uff0c\u53ef\u4ee5\u901a\u8fc7 ",(0,s.jsx)(t.code,{children:"bash systemctl list-unit-files"})," \u8fd9\u4e2a\u547d\u4ee4\u67e5\u770b\u6240\u6709\u670d\u52a1\uff0c\u6572\u4e0b\u56de\u8f66\u540e\u6253\u5370\u51fa\u6765\u8fd9\u4e48\u4e00\u5806\u73a9\u5e94\u513f\uff1a"]}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.img,{alt:"services",src:n(3120).A+"",title:"services",width:"454",height:"474"})}),"\n",(0,s.jsxs)(t.p,{children:["service \u7684 ",(0,s.jsx)(t.code,{children:"disabled"})," \u548c ",(0,s.jsx)(t.code,{children:"enabled"})," \u72b6\u6001\u90fd\u597d\u7406\u89e3\uff0c",(0,s.jsx)(t.code,{children:"static"})," \u662f\u4e2a\u5565\uff1f\u5728",(0,s.jsx)(t.a,{href:"https://bbs.archlinux.org/viewtopic.php?id=147964",title:"systemd 'static' unit file state",children:"\u4e0d\u5b58\u5728\u7684\u7f51\u7ad9"}),"\u4e0a\u4e00\u987f\u67e5\u627e\uff0c\u627e\u5230\u5982\u4e0b\u8fd9\u756a\u89e3\u91ca\uff1a"]}),"\n",(0,s.jsxs)(t.blockquote,{children:["\n",(0,s.jsx)(t.p,{children:'"static" means "enabled because something else wants it". Think by analogy to pacman\'s package install reasons:'}),"\n",(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsx)(t.li,{children:"enabled :: explicitly installed"}),"\n",(0,s.jsx)(t.li,{children:"static :: installed as dependency"}),"\n",(0,s.jsx)(t.li,{children:"disabled :: not installed"}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(t.p,{children:["\u610f\u601d\u662f\uff0c\u72b6\u6001\u4e3a ",(0,s.jsx)(t.code,{children:"static"})," \u7684\u670d\u52a1\uff0c\u662f\u4f5c\u4e3a\u522b\u7684\u670d\u52a1\u7684\u4f9d\u8d56\u800c\u5b58\u5728\u3002"]})]})}function u(e={}){const{wrapper:t}={...(0,l.R)(),...e.components};return t?(0,s.jsx)(t,{...e,children:(0,s.jsx)(d,{...e})}):d(e)}},8453:(e,t,n)=>{n.d(t,{R:()=>a,x:()=>r});var i=n(6540);const s={},l=i.createContext(s);function a(e){const t=i.useContext(l);return i.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function r(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:a(e.components),i.createElement(l.Provider,{value:t},e.children)}}}]);