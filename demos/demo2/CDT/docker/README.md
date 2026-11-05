# Running CDT in Docker

This directory contains a container setup for running the CDT GUI on a
**headless Linux server** and viewing it from your own computer over SSH.

There are two ways to see the GUI:

| Mode | How it works | Needs on your computer | Best for |
|------|--------------|------------------------|----------|
| **X11** (default) | `ssh -Y` forwards the CDT windows to your local screen | An X server (see below) | Linux desktops, fast networks |
| **VNC** | CDT runs on a virtual desktop inside the container; you view it in a web browser through an SSH tunnel | Only a web browser | Windows/macOS without an X server, slow or high-latency links |

Files:

- `../Dockerfile`: image definition (based on `rocker/r2u:noble`, works on amd64 and arm64)
- `entrypoint.sh`: starts CDT in the selected mode inside the container
- `../cdt-docker.sh`: launcher script you run on the server

---

## 1. Server setup (one time)

On the server you need Docker, and your user must be able to run it
(`docker ps` should work without `sudo`). If it doesn't, ask an admin to add
you to the `docker` group, then log out and back in.

Clone the repository and build the image:

```bash
git clone https://github.com/amarchakitus/CDT.git
cd CDT
./cdt-docker.sh build
```

The first build takes a few minutes. Rebuild after pulling new CDT code.

For X11 mode, the server's SSH daemon must allow X11 forwarding. Check with:

```bash
grep -i '^X11Forwarding' /etc/ssh/sshd_config     # should print: X11Forwarding yes
```

`xauth` must also be installed on the server (`sudo apt install xauth`).

---

## 2. X11 mode (CDT windows on your local screen)

### 2a. Install an X server on your computer

| Your OS | What to install |
|---------|-----------------|
| Linux (desktop) | Nothing, you already have one |
| macOS | [XQuartz](https://www.xquartz.org), then log out and back in |
| Windows | [MobaXterm](https://mobaxterm.mobatek.net) (includes SSH + X server), or [VcXsrv](https://sourceforge.net/projects/vcxsrv/) together with OpenSSH |

### 2b. Connect with X11 forwarding

From a terminal on your computer:

```bash
ssh -Y user@server
```

`-Y` enables trusted X11 forwarding. If your client doesn't support `-Y`, use
`-X`. To make it permanent, add this to `~/.ssh/config` on your computer:

```
Host myserver
    HostName server.example.org
    User user
    ForwardX11 yes
    ForwardX11Trusted yes
    Compression yes
```

then just run `ssh myserver`.

- **MobaXterm:** create an SSH session; "X11-Forwarding" is on by default.
- **Windows + VcXsrv:** start XLaunch ("Multiple windows", tick "Disable access
  control"), then in PowerShell run `$env:DISPLAY="localhost:0.0"` followed by
  `ssh -Y user@server`.

Check that forwarding works. On the server, this should print something
like `localhost:10.0`:

```bash
echo $DISPLAY
```

### 2c. Start CDT

```bash
cd /path/to/your/data      # this folder becomes CDT's working directory
~/CDT/cdt-docker.sh
```

The CDT main window opens on your screen. Closing it stops the container.

---

## 3. VNC mode (CDT in your web browser)

No X server is needed on your computer. CDT runs on a virtual desktop inside
the container, which is served as a web page (noVNC) on the server's port
6080, reachable **only from the server itself**. You reach it with an SSH tunnel.

**Step 1: open an SSH session with a tunnel.** From your computer:

```bash
ssh -L 6080:localhost:6080 user@server
```

(MobaXterm/PuTTY: add a local port forward, source port `6080`, destination
`localhost:6080`.)

**Step 2: start CDT in VNC mode** in that SSH session:

```bash
cd /path/to/your/data
~/CDT/cdt-docker.sh vnc
```

**Step 3: open the GUI.** On your computer, browse to:

```
http://localhost:6080/vnc.html?autoconnect=1&resize=scale
```

If you close the CDT window it restarts automatically. To stop it, press
`Ctrl-C` in the SSH session (or `docker stop` the container).

If port 6080 is already taken (for example when several people share the
server), pick another one and use it in both places:

```bash
ssh -L 6090:localhost:6090 user@server
NOVNC_PORT=6090 ~/CDT/cdt-docker.sh vnc
```

> **Security note:** the VNC desktop has no password. It is only bound to
> `127.0.0.1` on the server, but any user who can log in to the server can
> connect to it while it is running. Don't use VNC mode on a server you share
> with untrusted users.

To keep CDT running after you disconnect, start it inside `tmux` or `screen`;
later you can reconnect the tunnel and reload the browser page.

---

## 4. Where your files go

| On the server | Inside the container | Contents |
|---------------|----------------------|----------|
| Current directory (or `$CDT_DATA`) | `/data` | CDT working directory: your input data and outputs |
| `~/.cdt-docker` (or `$CDT_HOME`) | `/home/cdt` | CDT configuration (`Documents/CDT_Local_Config`), kept between runs |

The container runs as your own user, so files it creates are owned by you.
Only these two directories are visible to CDT. To work on data elsewhere,
point `CDT_DATA` at it:

```bash
CDT_DATA=/srv/climate/stations ~/CDT/cdt-docker.sh
```

## 5. Options

All options are environment variables set when running `cdt-docker.sh`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CDT_DATA` | current directory | Host folder mounted as `/data` |
| `CDT_HOME` | `~/.cdt-docker` | Host folder for persistent CDT settings |
| `CDT_IMAGE` | `cdt:latest` | Image to run |
| `NOVNC_PORT` | `6080` | Server port for VNC mode |
| `VNC_GEOMETRY` | `1600x900` | Virtual desktop size for VNC mode |

Other commands:

```bash
./cdt-docker.sh build      # (re)build the image
./cdt-docker.sh R          # R console inside the container (library(CDT) works)
```

---

## 6. Troubleshooting

**`No DISPLAY` when starting in X11 mode.** Your SSH session has no X11
forwarding. Reconnect with `ssh -Y`, make sure your local X server is running,
and confirm `X11Forwarding yes` on the server. `ssh -v -Y user@server` shows
`X11 forwarding request failed` if the server refuses it.

**`Can't open display` / `Authorization required`.** The launcher copies your
session's X cookie for the container. This fails if `xauth` is missing on the
server, or if you changed user with `sudo`/`su` after logging in. Run the
launcher as the user you logged in as.

**The GUI is very slow in X11 mode.** Tk redraws a lot over the network.
Add `-C` (compression) to `ssh`, or switch to VNC mode, which is usually much
faster over long-distance connections.

**`permission denied ... docker.sock`.** Your user is not in the `docker`
group (see section 1).

**Browser shows "Failed to connect" in VNC mode.** Check that the container is
still running, that the tunnel's port matches `NOVNC_PORT`, and that nothing
else on your computer is already using that port.

**Resetting CDT's configuration.** Delete `~/.cdt-docker/Documents/CDT_Local_Config`;
it is recreated with defaults on the next start.

### Note for maintainers

CDT works out its local configuration path (`~/Documents/CDT_Local_Config`)
when the package is **installed**, not when it runs. The image therefore
installs CDT with `HOME=/home/cdt` and always runs with that same `HOME`.
Keep the two the same if you change the Dockerfile.
