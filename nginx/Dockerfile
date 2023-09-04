# Pull nginx image from Docker Hub
FROM nginx

# Remove the default nginx.conf and replace with our own
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d