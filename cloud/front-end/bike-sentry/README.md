TO DEPLOY front-end code

1. yarn build (this will make the static files)
2. open CMD in project root
3. run `firebase login:ci`
4. Note token in console (we will need that token in future steps)
5. firebase deploy --token "TOKEN GOES HERE" (but you dont need quotes)
