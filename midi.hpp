#ifndef MIDI_HPP__
#define MIDI_HPP__

#include <vector>
#include <stddef.h>
#include <stdint.h>

namespace MIDI
{
   struct Event
   {
      bool kon;
      uint8_t note;
      uint8_t velocity;
      uint32_t frame;
   };

   class File
   {
      public:
         File(double fps, const char *path);
         File() = default;

         unsigned get_num_tracks() const;
         void step();
         void reset();
         bool eof() const { return is_eof; }

         const std::vector<Event> &get_events(unsigned track) const;
         double get_beat() const;
         void seek(unsigned tick);

      private:
         struct Track
         {
            std::vector<Event> current_events;
            std::vector<Event> all_events;
            unsigned events_ptr = 0;
         };
         std::vector<Track> tracks;
         unsigned units_per_beat = 0;
         double time_per_beat = 0.0;
         double fps;
         unsigned frame = 0;
         bool is_eof = false;

         void parse_track(Track &track, const uint8_t *data, size_t size);
   };
}

#endif
