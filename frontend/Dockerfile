# --- Stage 1: build the SPA ----------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Copy manifests first so dependency install stays cached across source edits.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- Stage 2: serve static assets ----------------------------------------
FROM nginx:1.27-alpine AS runtime

RUN apk add --no-cache curl

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD curl -fsS http://localhost/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
