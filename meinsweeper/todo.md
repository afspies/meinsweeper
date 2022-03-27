# ToDos
* Add logging option for transferring of files - i.e. use -p mode from rsync and add this as a separate process during initialization (and finalization) stage of a producer
* Restart partial rsync in the event of a failure during copying
* Add a target blacklist (rather than just recyling functioning targets. This is to facilitate:
    * Periodically re-check blacklisted targets