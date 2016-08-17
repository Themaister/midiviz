#include "midi.hpp"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdexcept>
#include <memory>

using namespace std;

namespace MIDI
{
   struct FILEDeleter
   {
      void operator()(FILE *file)
      {
         if (file)
            fclose(file);
      }
   };

   static void read_tag(FILE *file, const char *tag)
   {
      char tmp[4];
      if (fread(tmp, sizeof(tmp), 1, file) != 1)
         throw runtime_error("Failed to parse MIDI file.");
      if (memcmp(tmp, tag, 4))
         throw runtime_error("Failed to parse MIDI file.");
   }

   static uint32_t read_u32(FILE *file)
   {
      uint8_t v[4];
      if (fread(v, sizeof(v), 1, file) != 1)
         throw runtime_error("Failed to parse MIDI file.");
      return (v[0] << 24) | (v[1] << 16) | (v[2] << 8) | v[3];
   }

   static uint16_t read_u16(FILE *file)
   {
      uint8_t v[2];
      if (fread(v, sizeof(v), 1, file) != 1)
         throw runtime_error("Failed to parse MIDI file.");
      return (v[0] << 8) | v[1];
   }

   File::File(double fps, const char *path)
      : fps(fps)
   {
      unique_ptr<FILE, FILEDeleter> file(fopen(path, "rb"));
      if (!file)
         throw runtime_error("Failed to open MIDI file.");

      read_tag(file.get(), "MThd");

      if (read_u32(file.get()) != 6)
         throw runtime_error("Failed to parse MIDI file.");

      if (read_u16(file.get()) != 1)
         throw runtime_error("Expected multitrack file.");

      unsigned num_tracks = read_u16(file.get());
      tracks.resize(num_tracks);
      units_per_beat = read_u16(file.get());

      for (unsigned i = 0; i < num_tracks; i++)
      {
         vector<uint8_t> blob;
         read_tag(file.get(), "MTrk");
         unsigned length = read_u32(file.get());

         blob.resize(length);
         if (fread(blob.data(), blob.size(), 1, file.get()) != 1)
            throw runtime_error("Failed to parse MIDI track.");

         parse_track(tracks[i], blob.data(), blob.size());
      }
   }

   void File::parse_track(Track &track, const uint8_t *data, size_t)
   {
      uint32_t current = 0;
      uint8_t running = 0;

      auto read_varint = [&data]() -> uint32_t {
         uint32_t v = 0;
         uint8_t inval;
         do
         {
            inval = *data++;
            v = (v << 7) | (inval & 0x7f);
         } while (inval & 0x80);
         return v;
      };

      for (;;)
      {
         uint32_t delta_time = read_varint();
         current += delta_time;
         uint8_t type = *data >> 4;

         if (type < 0x8) // Running status
            type = running;
         else
            data++;

         if (type == 0x9) // KON
         {
            uint8_t key = *data++;
            uint8_t vel = *data++;
            running = type;

            double time = time_per_beat * double(current) / double(units_per_beat);
            uint32_t frame = uint32_t(round(time * fps));
            track.all_events.push_back({ true, key, vel, frame });
         }
         else if (type == 0x8) // KOF
         {
            uint8_t key = *data++;
            uint8_t vel = *data++;
            running = type;

            double time = time_per_beat * double(current) / double(units_per_beat);
            uint32_t frame = uint32_t(round(time * fps));
            track.all_events.push_back({ true, key, vel, frame });
         }
         else if (type == 0xa || type == 0xb || type == 0xe)
            data += 2;
         else if (type == 0xc || type == 0xd)
            data += 1;
         else if (data[-1] == 0xff) // Meta
         {
            uint8_t type = *data++;
            uint32_t len = read_varint();

            switch (type)
            {
               case 0x2f: // EoT
                  return;

               case 0x51: // Set Tempo
                  time_per_beat = double((data[0] << 16) | (data[1] << 8) | (data[2] << 0)) / 1e6;
                  break;

               case 0x58: // Time signature
                  //clocks_per_tick = data[2];
                  break;
            }

            data += len;
         }
         else
            throw runtime_error("Failed to parse MIDI file.");
      }
   }

   unsigned File::get_num_tracks() const
   {
      return tracks.size();
   }

   void File::reset()
   {
      for (auto &track : tracks)
         track.events_ptr = 0;
      frame = 0;
   }

   void File::step()
   {
      is_eof = true;
      for (auto &track : tracks)
      {
         track.current_events.clear();
         while (track.events_ptr < track.all_events.size() &&
                frame >= track.all_events[track.events_ptr].frame)
         {
            track.current_events.push_back(track.all_events[track.events_ptr]);
            track.events_ptr++;
            is_eof = false;
         }

         if (track.events_ptr < track.all_events.size())
            is_eof = false;
      }
      frame++;
   }

   double File::get_beat() const
   {
      return double(frame) / (fps * time_per_beat);
   }

   void File::seek(unsigned count)
   {
      if (count >= frame)
      {
         for (unsigned i = frame; i < count; i++)
            step();
      }
      else
      {
         // Crappy way of doing this.
         reset();
         for (unsigned i = 0; i < count; i++)
            step();
      }
   }

   const vector<Event> &File::get_events(unsigned track) const
   {
      return tracks[track].current_events;
   }
}
