# maia2-js

This repository provides the Maia2 chess model in ONNX format along with a small Next.js example app.

## Example App

The `example/` folder contains a Next.js project that demonstrates how to load the ONNX model in the browser. After installing dependencies you can run the development server:

```bash
cd example
npm install
npm run dev
```

Open <http://localhost:3000> to view the app.

## Deploy to Vercel

You can deploy the demo with one click using the button below:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-user/maia2-js)

The link will clone this repository into a new Vercel project and automatically build the Next.js app. Replace `your-user` with your GitHub account if you fork the repo first.

## Deploy to Netlify

If you prefer using [Netlify](https://www.netlify.com/), you can deploy from this repository as well:

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository-url=https://github.com/your-user/maia2-js)

Netlify reads the `netlify.toml` file and builds the app from the `example/` directory.

## Deploy to Cloudflare Pages

[Cloudflare Pages](https://pages.cloudflare.com/) also offers free hosting. When creating a project, set the root directory to `example` and follow [Cloudflare's Next.js guide](https://developers.cloudflare.com/pages/framework-guides/deploy-a-nextjs-site/) for more details.
