"use strict";(()=>{var e={};e.id=386,e.ids=[386],e.modules={517:e=>{e.exports=require("next/dist/compiled/next-server/app-route.runtime.prod.js")},2100:(e,t,r)=>{r.r(t),r.d(t,{headerHooks:()=>m,originalPathname:()=>x,patchFetch:()=>y,requestAsyncStorage:()=>h,routeModule:()=>c,serverHooks:()=>g,staticGenerationAsyncStorage:()=>f,staticGenerationBailout:()=>b});var n={};r.r(n),r.d(n,{POST:()=>u});var a=r(5419),o=r(9108),i=r(9678),s=r(8070);let l="info@zyoralabs.com",d={general:"General Inquiry",support:"Technical Support",enterprise:"Enterprise / Partnership",feedback:"Feedback / Suggestions",other:"Other"};async function p(e,t,r){try{let n=await fetch("https://api.zeptomail.in/v1.1/email",{method:"POST",headers:{Authorization:"Zoho-enczapikey PHtE6r0PQO/j2WEr+kBT7KC7H5P3Z98o+uxgfwAW444XWKIGSU1Vr9ktlGC/oxwsU/BCHP+byIw7sO+c4L/TLWu4YGYfWmqyqK3sx/VYSPOZsbq6x00btlQScULdUo7pc99o0ifVud/cNA==","Content-Type":"application/json",Accept:"application/json"},body:JSON.stringify({from:{address:"info@zyoralabs.com",name:"ZSE by Zlabs"},to:[{email_address:{address:e,name:e===l?"ZSE Admin":void 0}}],subject:t,htmlbody:r})});if(!n.ok){let e=await n.text();return console.error("ZeptoMail API error:",e),!1}return!0}catch(e){return console.error("Error sending email:",e),!1}}async function u(e){try{let t=await e.json();if(!t.name||!t.email||!t.subject||!t.message)return s.Z.json({success:!1,error:"All fields are required"},{status:400});if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(t.email))return s.Z.json({success:!1,error:"Invalid email address"},{status:400});let r=d[t.subject]||t.subject,n=await p(l,`[ZSE Contact] ${r} from ${t.name}`,function(e){let t=d[e.subject]||e.subject;return`
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #000000; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #000000;">
    <tr>
      <td align="center" style="padding: 40px 20px;">
        <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #0a0a0a; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
          <!-- Header -->
          <tr>
            <td style="padding: 40px 40px 20px 40px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);">
              <h1 style="margin: 0; color: #c0ff71; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">
                New Contact Form Submission
              </h1>
              <p style="margin: 10px 0 0 0; color: rgba(255,255,255,0.5); font-size: 14px;">
                ZSE Website Contact Form
              </p>
            </td>
          </tr>
          
          <!-- Content -->
          <tr>
            <td style="padding: 30px 40px;">
              <!-- Sender Info -->
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin-bottom: 24px;">
                <tr>
                  <td style="padding: 20px; background-color: rgba(192,255,113,0.05); border-radius: 12px; border: 1px solid rgba(192,255,113,0.1);">
                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                      <tr>
                        <td width="50%" style="padding: 8px 0;">
                          <p style="margin: 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">From</p>
                          <p style="margin: 4px 0 0 0; color: #ffffff; font-size: 16px; font-weight: 600;">${e.name}</p>
                        </td>
                        <td width="50%" style="padding: 8px 0;">
                          <p style="margin: 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Email</p>
                          <p style="margin: 4px 0 0 0; color: #c0ff71; font-size: 16px;">
                            <a href="mailto:${e.email}" style="color: #c0ff71; text-decoration: none;">${e.email}</a>
                          </p>
                        </td>
                      </tr>
                      <tr>
                        <td colspan="2" style="padding: 8px 0;">
                          <p style="margin: 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Subject</p>
                          <p style="margin: 4px 0 0 0; color: #ffffff; font-size: 16px;">${t}</p>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
              
              <!-- Message -->
              <div style="margin-bottom: 24px;">
                <p style="margin: 0 0 12px 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Message</p>
                <div style="padding: 20px; background-color: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
                  <p style="margin: 0; color: rgba(255,255,255,0.85); font-size: 15px; line-height: 1.7; white-space: pre-wrap;">${e.message}</p>
                </div>
              </div>
              
              <!-- Reply Button -->
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                <tr>
                  <td align="center">
                    <a href="mailto:${e.email}?subject=Re: ${t}" style="display: inline-block; padding: 14px 32px; background-color: #c0ff71; color: #000000; font-size: 14px; font-weight: 600; text-decoration: none; border-radius: 8px;">
                      Reply to ${e.name}
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 24px 40px; border-top: 1px solid rgba(255,255,255,0.1); text-align: center;">
              <p style="margin: 0; color: rgba(255,255,255,0.4); font-size: 13px;">
                This email was sent from the ZSE website contact form
              </p>
              <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.3); font-size: 12px;">
                \xa9 ${new Date().getFullYear()} Zyora Labs. All rights reserved.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
`}(t)),a=await p(t.email,"Thank you for contacting ZSE - We received your message!",function(e){let t=d[e.subject]||e.subject;return`
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #000000; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #000000;">
    <tr>
      <td align="center" style="padding: 40px 20px;">
        <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #0a0a0a; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
          <!-- Header with Logo -->
          <tr>
            <td style="padding: 40px 40px 30px 40px; text-align: center;">
              <div style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, rgba(192,255,113,0.15) 0%, rgba(192,255,113,0.05) 100%); border-radius: 12px; border: 1px solid rgba(192,255,113,0.2);">
                <span style="color: #c0ff71; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">zLLM</span>
                <span style="color: rgba(255,255,255,0.6); font-size: 24px; font-weight: 300;"> | </span>
                <span style="color: #ffffff; font-size: 24px; font-weight: 700;">ZSE</span>
              </div>
            </td>
          </tr>
          
          <!-- Main Content -->
          <tr>
            <td style="padding: 0 40px 40px 40px;">
              <!-- Greeting -->
              <h1 style="margin: 0 0 20px 0; color: #ffffff; font-size: 28px; font-weight: 700; text-align: center;">
                Thank you for reaching out!
              </h1>
              
              <p style="margin: 0 0 24px 0; color: rgba(255,255,255,0.7); font-size: 16px; line-height: 1.7; text-align: center;">
                Hi <strong style="color: #ffffff;">${e.name}</strong>, we've received your message and our team will get back to you as soon as possible.
              </p>
              
              <!-- Summary Card -->
              <div style="padding: 24px; background-color: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 32px;">
                <p style="margin: 0 0 16px 0; color: rgba(255,255,255,0.5); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Your Message Summary</p>
                
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                  <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
                      <span style="color: rgba(255,255,255,0.5); font-size: 14px;">Subject:</span>
                      <span style="color: #ffffff; font-size: 14px; margin-left: 8px;">${t}</span>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding: 12px 0 0 0;">
                      <p style="margin: 0 0 8px 0; color: rgba(255,255,255,0.5); font-size: 14px;">Message:</p>
                      <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 14px; line-height: 1.6; white-space: pre-wrap;">${e.message.substring(0,300)}${e.message.length>300?"...":""}</p>
                    </td>
                  </tr>
                </table>
              </div>
              
              <!-- Response Time -->
              <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(192,255,113,0.08) 0%, rgba(192,255,113,0.02) 100%); border-radius: 12px; border: 1px solid rgba(192,255,113,0.15); margin-bottom: 32px;">
                <p style="margin: 0; color: #c0ff71; font-size: 14px; font-weight: 600;">
                  ⚡ Expected Response Time: Within 24-48 hours
                </p>
              </div>
              
              <!-- Quick Links -->
              <p style="margin: 0 0 16px 0; color: rgba(255,255,255,0.5); font-size: 14px; text-align: center;">
                While you wait, explore our resources:
              </p>
              
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                <tr>
                  <td align="center">
                    <table role="presentation" cellspacing="0" cellpadding="0">
                      <tr>
                        <td style="padding: 0 8px;">
                          <a href="https://zse.dev/docs" style="display: inline-block; padding: 12px 20px; background-color: rgba(255,255,255,0.05); color: #ffffff; font-size: 13px; font-weight: 500; text-decoration: none; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                            📚 Documentation
                          </a>
                        </td>
                        <td style="padding: 0 8px;">
                          <a href="https://github.com/zse/zse" style="display: inline-block; padding: 12px 20px; background-color: rgba(255,255,255,0.05); color: #ffffff; font-size: 13px; font-weight: 500; text-decoration: none; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                            ⭐ GitHub
                          </a>
                        </td>
                        <td style="padding: 0 8px;">
                          <a href="https://discord.gg/f9JKreJA" style="display: inline-block; padding: 12px 20px; background-color: rgba(255,255,255,0.05); color: #ffffff; font-size: 13px; font-weight: 500; text-decoration: none; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                            💬 Discord
                          </a>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 24px 40px; border-top: 1px solid rgba(255,255,255,0.1); text-align: center;">
              <p style="margin: 0 0 8px 0; color: rgba(255,255,255,0.6); font-size: 14px;">
                <strong style="color: #c0ff71;">ZSE</strong> by Zyora Labs
              </p>
              <p style="margin: 0 0 4px 0; color: rgba(255,255,255,0.4); font-size: 13px;">
                Fast LLM Inference Engine | 3.9s Cold Starts
              </p>
              <p style="margin: 12px 0 0 0; color: rgba(255,255,255,0.3); font-size: 12px;">
                \xa9 ${new Date().getFullYear()} Zyora Labs, Tamil Nadu, India. All rights reserved.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
`}(t));if(!n)return s.Z.json({success:!1,error:"Failed to send message. Please try again."},{status:500});return s.Z.json({success:!0,message:"Message sent successfully",autoReplySent:a})}catch(e){return console.error("Contact form error:",e),s.Z.json({success:!1,error:"An unexpected error occurred"},{status:500})}}let c=new a.AppRouteRouteModule({definition:{kind:o.x.APP_ROUTE,page:"/api/contact/route",pathname:"/api/contact",filename:"route",bundlePath:"app/api/contact/route"},resolvedPagePath:"/Users/redfoxhotels/zse/website/src/app/api/contact/route.ts",nextConfigOutput:"",userland:n}),{requestAsyncStorage:h,staticGenerationAsyncStorage:f,serverHooks:g,headerHooks:m,staticGenerationBailout:b}=c,x="/api/contact/route";function y(){return(0,i.patchFetch)({serverHooks:g,staticGenerationAsyncStorage:f})}},7347:e=>{var t=Object.defineProperty,r=Object.getOwnPropertyDescriptor,n=Object.getOwnPropertyNames,a=Object.prototype.hasOwnProperty,o={};function i(e){var t;let r=["path"in e&&e.path&&`Path=${e.path}`,"expires"in e&&(e.expires||0===e.expires)&&`Expires=${("number"==typeof e.expires?new Date(e.expires):e.expires).toUTCString()}`,"maxAge"in e&&"number"==typeof e.maxAge&&`Max-Age=${e.maxAge}`,"domain"in e&&e.domain&&`Domain=${e.domain}`,"secure"in e&&e.secure&&"Secure","httpOnly"in e&&e.httpOnly&&"HttpOnly","sameSite"in e&&e.sameSite&&`SameSite=${e.sameSite}`,"priority"in e&&e.priority&&`Priority=${e.priority}`].filter(Boolean);return`${e.name}=${encodeURIComponent(null!=(t=e.value)?t:"")}; ${r.join("; ")}`}function s(e){let t=new Map;for(let r of e.split(/; */)){if(!r)continue;let e=r.indexOf("=");if(-1===e){t.set(r,"true");continue}let[n,a]=[r.slice(0,e),r.slice(e+1)];try{t.set(n,decodeURIComponent(null!=a?a:"true"))}catch{}}return t}function l(e){var t,r;if(!e)return;let[[n,a],...o]=s(e),{domain:i,expires:l,httponly:u,maxage:c,path:h,samesite:f,secure:g,priority:m}=Object.fromEntries(o.map(([e,t])=>[e.toLowerCase(),t]));return function(e){let t={};for(let r in e)e[r]&&(t[r]=e[r]);return t}({name:n,value:decodeURIComponent(a),domain:i,...l&&{expires:new Date(l)},...u&&{httpOnly:!0},..."string"==typeof c&&{maxAge:Number(c)},path:h,...f&&{sameSite:d.includes(t=(t=f).toLowerCase())?t:void 0},...g&&{secure:!0},...m&&{priority:p.includes(r=(r=m).toLowerCase())?r:void 0}})}((e,r)=>{for(var n in r)t(e,n,{get:r[n],enumerable:!0})})(o,{RequestCookies:()=>u,ResponseCookies:()=>c,parseCookie:()=>s,parseSetCookie:()=>l,stringifyCookie:()=>i}),e.exports=((e,o,i,s)=>{if(o&&"object"==typeof o||"function"==typeof o)for(let i of n(o))a.call(e,i)||void 0===i||t(e,i,{get:()=>o[i],enumerable:!(s=r(o,i))||s.enumerable});return e})(t({},"__esModule",{value:!0}),o);var d=["strict","lax","none"],p=["low","medium","high"],u=class{constructor(e){this._parsed=new Map,this._headers=e;let t=e.get("cookie");if(t)for(let[e,r]of s(t))this._parsed.set(e,{name:e,value:r})}[Symbol.iterator](){return this._parsed[Symbol.iterator]()}get size(){return this._parsed.size}get(...e){let t="string"==typeof e[0]?e[0]:e[0].name;return this._parsed.get(t)}getAll(...e){var t;let r=Array.from(this._parsed);if(!e.length)return r.map(([e,t])=>t);let n="string"==typeof e[0]?e[0]:null==(t=e[0])?void 0:t.name;return r.filter(([e])=>e===n).map(([e,t])=>t)}has(e){return this._parsed.has(e)}set(...e){let[t,r]=1===e.length?[e[0].name,e[0].value]:e,n=this._parsed;return n.set(t,{name:t,value:r}),this._headers.set("cookie",Array.from(n).map(([e,t])=>i(t)).join("; ")),this}delete(e){let t=this._parsed,r=Array.isArray(e)?e.map(e=>t.delete(e)):t.delete(e);return this._headers.set("cookie",Array.from(t).map(([e,t])=>i(t)).join("; ")),r}clear(){return this.delete(Array.from(this._parsed.keys())),this}[Symbol.for("edge-runtime.inspect.custom")](){return`RequestCookies ${JSON.stringify(Object.fromEntries(this._parsed))}`}toString(){return[...this._parsed.values()].map(e=>`${e.name}=${encodeURIComponent(e.value)}`).join("; ")}},c=class{constructor(e){var t,r,n;this._parsed=new Map,this._headers=e;let a=null!=(n=null!=(r=null==(t=e.getSetCookie)?void 0:t.call(e))?r:e.get("set-cookie"))?n:[];for(let e of Array.isArray(a)?a:function(e){if(!e)return[];var t,r,n,a,o,i=[],s=0;function l(){for(;s<e.length&&/\s/.test(e.charAt(s));)s+=1;return s<e.length}for(;s<e.length;){for(t=s,o=!1;l();)if(","===(r=e.charAt(s))){for(n=s,s+=1,l(),a=s;s<e.length&&"="!==(r=e.charAt(s))&&";"!==r&&","!==r;)s+=1;s<e.length&&"="===e.charAt(s)?(o=!0,s=a,i.push(e.substring(t,n)),t=s):s=n+1}else s+=1;(!o||s>=e.length)&&i.push(e.substring(t,e.length))}return i}(a)){let t=l(e);t&&this._parsed.set(t.name,t)}}get(...e){let t="string"==typeof e[0]?e[0]:e[0].name;return this._parsed.get(t)}getAll(...e){var t;let r=Array.from(this._parsed.values());if(!e.length)return r;let n="string"==typeof e[0]?e[0]:null==(t=e[0])?void 0:t.name;return r.filter(e=>e.name===n)}has(e){return this._parsed.has(e)}set(...e){let[t,r,n]=1===e.length?[e[0].name,e[0].value,e[0]]:e,a=this._parsed;return a.set(t,function(e={name:"",value:""}){return"number"==typeof e.expires&&(e.expires=new Date(e.expires)),e.maxAge&&(e.expires=new Date(Date.now()+1e3*e.maxAge)),(null===e.path||void 0===e.path)&&(e.path="/"),e}({name:t,value:r,...n})),function(e,t){for(let[,r]of(t.delete("set-cookie"),e)){let e=i(r);t.append("set-cookie",e)}}(a,this._headers),this}delete(...e){let[t,r,n]="string"==typeof e[0]?[e[0]]:[e[0].name,e[0].path,e[0].domain];return this.set({name:t,path:r,domain:n,value:"",expires:new Date(0)})}[Symbol.for("edge-runtime.inspect.custom")](){return`ResponseCookies ${JSON.stringify(Object.fromEntries(this._parsed))}`}toString(){return[...this._parsed.values()].map(i).join("; ")}}},5419:(e,t,r)=>{e.exports=r(517)},8070:(e,t,r)=>{Object.defineProperty(t,"Z",{enumerable:!0,get:function(){return n.NextResponse}});let n=r(457)},514:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"NextURL",{enumerable:!0,get:function(){return p}});let n=r(737),a=r(5418),o=r(283),i=r(3588),s=/(?!^https?:\/\/)(127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}|\[::1\]|localhost)/;function l(e,t){return new URL(String(e).replace(s,"localhost"),t&&String(t).replace(s,"localhost"))}let d=Symbol("NextURLInternal");class p{constructor(e,t,r){let n,a;"object"==typeof t&&"pathname"in t||"string"==typeof t?(n=t,a=r||{}):a=r||t||{},this[d]={url:l(e,n??a.base),options:a,basePath:""},this.analyze()}analyze(){var e,t,r,a,s;let l=(0,i.getNextPathnameInfo)(this[d].url.pathname,{nextConfig:this[d].options.nextConfig,parseData:!0,i18nProvider:this[d].options.i18nProvider}),p=(0,o.getHostname)(this[d].url,this[d].options.headers);this[d].domainLocale=this[d].options.i18nProvider?this[d].options.i18nProvider.detectDomainLocale(p):(0,n.detectDomainLocale)(null==(t=this[d].options.nextConfig)?void 0:null==(e=t.i18n)?void 0:e.domains,p);let u=(null==(r=this[d].domainLocale)?void 0:r.defaultLocale)||(null==(s=this[d].options.nextConfig)?void 0:null==(a=s.i18n)?void 0:a.defaultLocale);this[d].url.pathname=l.pathname,this[d].defaultLocale=u,this[d].basePath=l.basePath??"",this[d].buildId=l.buildId,this[d].locale=l.locale??u,this[d].trailingSlash=l.trailingSlash}formatPathname(){return(0,a.formatNextPathnameInfo)({basePath:this[d].basePath,buildId:this[d].buildId,defaultLocale:this[d].options.forceLocale?void 0:this[d].defaultLocale,locale:this[d].locale,pathname:this[d].url.pathname,trailingSlash:this[d].trailingSlash})}formatSearch(){return this[d].url.search}get buildId(){return this[d].buildId}set buildId(e){this[d].buildId=e}get locale(){return this[d].locale??""}set locale(e){var t,r;if(!this[d].locale||!(null==(r=this[d].options.nextConfig)?void 0:null==(t=r.i18n)?void 0:t.locales.includes(e)))throw TypeError(`The NextURL configuration includes no locale "${e}"`);this[d].locale=e}get defaultLocale(){return this[d].defaultLocale}get domainLocale(){return this[d].domainLocale}get searchParams(){return this[d].url.searchParams}get host(){return this[d].url.host}set host(e){this[d].url.host=e}get hostname(){return this[d].url.hostname}set hostname(e){this[d].url.hostname=e}get port(){return this[d].url.port}set port(e){this[d].url.port=e}get protocol(){return this[d].url.protocol}set protocol(e){this[d].url.protocol=e}get href(){let e=this.formatPathname(),t=this.formatSearch();return`${this.protocol}//${this.host}${e}${t}${this.hash}`}set href(e){this[d].url=l(e),this.analyze()}get origin(){return this[d].url.origin}get pathname(){return this[d].url.pathname}set pathname(e){this[d].url.pathname=e}get hash(){return this[d].url.hash}set hash(e){this[d].url.hash=e}get search(){return this[d].url.search}set search(e){this[d].url.search=e}get password(){return this[d].url.password}set password(e){this[d].url.password=e}get username(){return this[d].url.username}set username(e){this[d].url.username=e}get basePath(){return this[d].basePath}set basePath(e){this[d].basePath=e.startsWith("/")?e:`/${e}`}toString(){return this.href}toJSON(){return this.href}[Symbol.for("edge-runtime.inspect.custom")](){return{href:this.href,origin:this.origin,protocol:this.protocol,username:this.username,password:this.password,host:this.host,hostname:this.hostname,port:this.port,pathname:this.pathname,search:this.search,searchParams:this.searchParams,hash:this.hash}}clone(){return new p(String(this),this[d].options)}}},3608:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),function(e,t){for(var r in t)Object.defineProperty(e,r,{enumerable:!0,get:t[r]})}(t,{RequestCookies:function(){return n.RequestCookies},ResponseCookies:function(){return n.ResponseCookies}});let n=r(7347)},457:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"NextResponse",{enumerable:!0,get:function(){return d}});let n=r(514),a=r(8670),o=r(3608),i=Symbol("internal response"),s=new Set([301,302,303,307,308]);function l(e,t){var r;if(null==e?void 0:null==(r=e.request)?void 0:r.headers){if(!(e.request.headers instanceof Headers))throw Error("request.headers must be an instance of Headers");let r=[];for(let[n,a]of e.request.headers)t.set("x-middleware-request-"+n,a),r.push(n);t.set("x-middleware-override-headers",r.join(","))}}class d extends Response{constructor(e,t={}){super(e,t),this[i]={cookies:new o.ResponseCookies(this.headers),url:t.url?new n.NextURL(t.url,{headers:(0,a.toNodeOutgoingHttpHeaders)(this.headers),nextConfig:t.nextConfig}):void 0}}[Symbol.for("edge-runtime.inspect.custom")](){return{cookies:this.cookies,url:this.url,body:this.body,bodyUsed:this.bodyUsed,headers:Object.fromEntries(this.headers),ok:this.ok,redirected:this.redirected,status:this.status,statusText:this.statusText,type:this.type}}get cookies(){return this[i].cookies}static json(e,t){let r=Response.json(e,t);return new d(r.body,r)}static redirect(e,t){let r="number"==typeof t?t:(null==t?void 0:t.status)??307;if(!s.has(r))throw RangeError('Failed to execute "redirect" on "response": Invalid status code');let n="object"==typeof t?t:{},o=new Headers(null==n?void 0:n.headers);return o.set("Location",(0,a.validateURL)(e)),new d(null,{...n,headers:o,status:r})}static rewrite(e,t){let r=new Headers(null==t?void 0:t.headers);return r.set("x-middleware-rewrite",(0,a.validateURL)(e)),l(t,r),new d(null,{...t,headers:r})}static next(e){let t=new Headers(null==e?void 0:e.headers);return t.set("x-middleware-next","1"),l(e,t),new d(null,{...e,headers:t})}}},8670:(e,t)=>{function r(e){let t=new Headers;for(let[r,n]of Object.entries(e))for(let e of Array.isArray(n)?n:[n])void 0!==e&&("number"==typeof e&&(e=e.toString()),t.append(r,e));return t}function n(e){var t,r,n,a,o,i=[],s=0;function l(){for(;s<e.length&&/\s/.test(e.charAt(s));)s+=1;return s<e.length}for(;s<e.length;){for(t=s,o=!1;l();)if(","===(r=e.charAt(s))){for(n=s,s+=1,l(),a=s;s<e.length&&"="!==(r=e.charAt(s))&&";"!==r&&","!==r;)s+=1;s<e.length&&"="===e.charAt(s)?(o=!0,s=a,i.push(e.substring(t,n)),t=s):s=n+1}else s+=1;(!o||s>=e.length)&&i.push(e.substring(t,e.length))}return i}function a(e){let t={},r=[];if(e)for(let[a,o]of e.entries())"set-cookie"===a.toLowerCase()?(r.push(...n(o)),t[a]=1===r.length?r[0]:r):t[a]=o;return t}function o(e){try{return String(new URL(String(e)))}catch(t){throw Error(`URL is malformed "${String(e)}". Please use only absolute URLs - https://nextjs.org/docs/messages/middleware-relative-urls`,{cause:t})}}Object.defineProperty(t,"__esModule",{value:!0}),function(e,t){for(var r in t)Object.defineProperty(e,r,{enumerable:!0,get:t[r]})}(t,{fromNodeOutgoingHttpHeaders:function(){return r},splitCookiesString:function(){return n},toNodeOutgoingHttpHeaders:function(){return a},validateURL:function(){return o}})},283:(e,t)=>{function r(e,t){let r;if((null==t?void 0:t.host)&&!Array.isArray(t.host))r=t.host.toString().split(":",1)[0];else{if(!e.hostname)return;r=e.hostname}return r.toLowerCase()}Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"getHostname",{enumerable:!0,get:function(){return r}})},737:(e,t)=>{function r(e,t,r){if(e)for(let o of(r&&(r=r.toLowerCase()),e)){var n,a;if(t===(null==(n=o.domain)?void 0:n.split(":",1)[0].toLowerCase())||r===o.defaultLocale.toLowerCase()||(null==(a=o.locales)?void 0:a.some(e=>e.toLowerCase()===r)))return o}}Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"detectDomainLocale",{enumerable:!0,get:function(){return r}})},3935:(e,t)=>{function r(e,t){let r;let n=e.split("/");return(t||[]).some(t=>!!n[1]&&n[1].toLowerCase()===t.toLowerCase()&&(r=t,n.splice(1,1),e=n.join("/")||"/",!0)),{pathname:e,detectedLocale:r}}Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"normalizeLocalePath",{enumerable:!0,get:function(){return r}})},8030:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"addLocale",{enumerable:!0,get:function(){return o}});let n=r(3495),a=r(7211);function o(e,t,r,o){if(!t||t===r)return e;let i=e.toLowerCase();return!o&&((0,a.pathHasPrefix)(i,"/api")||(0,a.pathHasPrefix)(i,"/"+t.toLowerCase()))?e:(0,n.addPathPrefix)(e,"/"+t)}},3495:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"addPathPrefix",{enumerable:!0,get:function(){return a}});let n=r(1955);function a(e,t){if(!e.startsWith("/")||!t)return e;let{pathname:r,query:a,hash:o}=(0,n.parsePath)(e);return""+t+r+a+o}},2348:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"addPathSuffix",{enumerable:!0,get:function(){return a}});let n=r(1955);function a(e,t){if(!e.startsWith("/")||!t)return e;let{pathname:r,query:a,hash:o}=(0,n.parsePath)(e);return""+r+t+a+o}},5418:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"formatNextPathnameInfo",{enumerable:!0,get:function(){return s}});let n=r(5545),a=r(3495),o=r(2348),i=r(8030);function s(e){let t=(0,i.addLocale)(e.pathname,e.locale,e.buildId?void 0:e.defaultLocale,e.ignorePrefix);return(e.buildId||!e.trailingSlash)&&(t=(0,n.removeTrailingSlash)(t)),e.buildId&&(t=(0,o.addPathSuffix)((0,a.addPathPrefix)(t,"/_next/data/"+e.buildId),"/"===e.pathname?"index.json":".json")),t=(0,a.addPathPrefix)(t,e.basePath),!e.buildId&&e.trailingSlash?t.endsWith("/")?t:(0,o.addPathSuffix)(t,"/"):(0,n.removeTrailingSlash)(t)}},3588:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"getNextPathnameInfo",{enumerable:!0,get:function(){return i}});let n=r(3935),a=r(7188),o=r(7211);function i(e,t){var r,i;let{basePath:s,i18n:l,trailingSlash:d}=null!=(r=t.nextConfig)?r:{},p={pathname:e,trailingSlash:"/"!==e?e.endsWith("/"):d};s&&(0,o.pathHasPrefix)(p.pathname,s)&&(p.pathname=(0,a.removePathPrefix)(p.pathname,s),p.basePath=s);let u=p.pathname;if(p.pathname.startsWith("/_next/data/")&&p.pathname.endsWith(".json")){let e=p.pathname.replace(/^\/_next\/data\//,"").replace(/\.json$/,"").split("/"),r=e[0];p.buildId=r,u="index"!==e[1]?"/"+e.slice(1).join("/"):"/",!0===t.parseData&&(p.pathname=u)}if(l){let e=t.i18nProvider?t.i18nProvider.analyze(p.pathname):(0,n.normalizeLocalePath)(p.pathname,l.locales);p.locale=e.detectedLocale,p.pathname=null!=(i=e.pathname)?i:p.pathname,!e.detectedLocale&&p.buildId&&(e=t.i18nProvider?t.i18nProvider.analyze(u):(0,n.normalizeLocalePath)(u,l.locales)).detectedLocale&&(p.locale=e.detectedLocale)}return p}},1955:(e,t)=>{function r(e){let t=e.indexOf("#"),r=e.indexOf("?"),n=r>-1&&(t<0||r<t);return n||t>-1?{pathname:e.substring(0,n?r:t),query:n?e.substring(r,t>-1?t:void 0):"",hash:t>-1?e.slice(t):""}:{pathname:e,query:"",hash:""}}Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"parsePath",{enumerable:!0,get:function(){return r}})},7211:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"pathHasPrefix",{enumerable:!0,get:function(){return a}});let n=r(1955);function a(e,t){if("string"!=typeof e)return!1;let{pathname:r}=(0,n.parsePath)(e);return r===t||r.startsWith(t+"/")}},7188:(e,t,r)=>{Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"removePathPrefix",{enumerable:!0,get:function(){return a}});let n=r(7211);function a(e,t){if(!(0,n.pathHasPrefix)(e,t))return e;let r=e.slice(t.length);return r.startsWith("/")?r:"/"+r}},5545:(e,t)=>{function r(e){return e.replace(/\/$/,"")||"/"}Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"removeTrailingSlash",{enumerable:!0,get:function(){return r}})}};var t=require("../../../webpack-runtime.js");t.C(e);var r=e=>t(t.s=e),n=t.X(0,[638],()=>r(2100));module.exports=n})();