# GPU cluster notes
[RCI SERVER - University of South Carolina](https://docs.google.com/document/d/1S4kpOkPnQeoAcIlQKFjZHeql1IsC4dw_oFTRXOGuGLI/edit?tab=t.0)

```bash
# Login
ssh -p<port> <user>@<address>.edu
```

- navigate to work for user, make project folder, then upload files from laptop/pc to server. tar file directly for quicker upload

```bash
tar -cd - data | pv | ssh -p <port> <user>@<addres>.edu "tar -xf - -C /work/<user>/park-data
```